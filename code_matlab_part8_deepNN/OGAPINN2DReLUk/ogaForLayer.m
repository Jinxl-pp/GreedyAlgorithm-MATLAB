function [errL2, Loss, parameters, approximator, Net] = ogaForLayer(Net, degree, pde, rectangle, quadRectangle)
%% initialization and activation functions
uk = @(x)(0);   % u
helmholtz_uk = @(x)(0); % Lap(u)+wn*(u)
wn = pde.wn^2;    % wave number of helmholtz eqn
mu = 1000;         % penalty parameter for BC
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);

%% largest value of L2 norm of neurons from last layer
x = linspace(rectangle(1,1),rectangle(1,2),1e2);
y = linspace(rectangle(2,1),rectangle(2,2),1e2);
[x,y] = meshgrid(x,y);
samples = [x(:)';y(:)'];
c = Net.preLayer(samples);
c = max(sqrt(sum(c.^2,1))) + 1;
square = [-c,c];
clear x y

%% quadrature information
% c     Guass quadreture points in 1D: discretize the interval and apply 
% c     2pts Gauss quadrature rule on each element.
% c     Guass quadreture points in 2D: 2D tensor product of 1D 
% c     Guass quadreture points in 3D: 3D tensor product of 1D 

pts = 4;
h = 1/quadRectangle;
[qdpts,wei] = rquadpts2d(rectangle,pts,quadRectangle);
numPts = size(qdpts,2);

%% quadrature on boudary
% c     For 1D, the quadrature is just a set of two points and uniform weights
% c     For 2D, the quadrature is reduced to 1D Gauss quadrature
[qdbc, weibc] = rquadpts2dbc(rectangle, pts, quadRectangle);
numPtsBc = size(qdbc,2);

%% the previous layer
layer = Net.layerIndicator;
nNeuron = Net.nNeuronsEach(layer+1);

% if the layer indicator is at the input layer
if Net.layerIndicator == 0
    inputDim = Net.inputDim;
    trainingSteps = nNeuron;
% if the layer indicator is at one of the hidden layers
else
    paramPreLayer = Net.parameters{layer};
    inputDim = size(paramPreLayer(:,1),1);
    trainingSteps = nNeuron - 2*degree;
end 
p = Net.preLayer(qdpts);
dxp = Net.dxpreLayer(qdpts);
dyp = Net.dypreLayer(qdpts);
lapp = Net.lappreLayer(qdpts);
pbc = Net.preLayer(qdbc);

%% auxiliary values
auxInfo.mu = mu; 
auxInfo.wn = wn;
auxInfo.deg = degree;
auxInfo.dim = inputDim;
auxInfo.quadRectangle = quadRectangle;

auxInfo.p = p;
auxInfo.dxp = dxp;
auxInfo.dyp = dyp;
auxInfo.lapp = lapp;
auxInfo.pbc = pbc;
auxInfo.f = pde.source(qdpts);
auxInfo.gD = pde.Dbc(qdbc);

f = (auxInfo.f.*wei)';
gD = (auxInfo.gD.*weibc)';

%% parameters initialization
W2 = zeros(trainingSteps,1);        
W1 = zeros(trainingSteps,inputDim);
Bias = zeros(trainingSteps,1);      
C = zeros(trainingSteps,numPts);
CBc = zeros(trainingSteps,numPtsBc);

%% Orthogonal greedy algorithm
errL2 = zeros(trainingSteps,1);
Loss = zeros(trainingSteps,1);
for k = 1:trainingSteps

    % c     The subproblem is key to the greedy algorithm, where the inner
    % c     products |(u,g) - (f,g)| should be maximized.
    % c     Part of the inner products can be computed in advance.

    auxInfo.uk = uk(qdpts); 
    auxInfo.huk = helmholtz_uk(qdpts);
    auxInfo.ukbc = uk(qdbc);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,numPts);
        auxInfo.huk = repmat(auxInfo.huk,1,numPts);
        auxInfo.ukbc = repmat(auxInfo.ukbc,1,numPtsBc);
    end
    
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration. Since |w|=1, only 
    % c     consider w = +-1.

    tic
    theta = optimalBasis(qdpts,wei,qdbc,weibc,square,auxInfo,h);
    fprintf('------------ the N = %d-th neuron ------------ \n', k);
    toc
    
    % c     A few steps of training are needed to reach a local minimum 
    % c     near the initial guess. Gradient descent, coordinate descent,
    % c     Newton's method, etc., can be applied to this problem.

    W1(k,:) = theta(1:end-1);
    Bias(k) = theta(end);
    w = W1(1:k,:);
    b = Bias(1:k);
    
    % c     Computation of the orthogonal projection. The Gram matrix 
    % c     G(i,j) = (Lap(g_i), Lap(g_j))
    % c     and the RHS vector 
    % c     b(j) = (f, g_j).
    % c     Also there is another Gram matrix from boundary condition
    % c     Gbc(i,j) = mu * (g_i, g_j)_{boundary}
    
    %%%
    ak = [W1(k,:),Bias(k)];
    B = [p; ones(1,size(qdpts,2))];
    BBc = [pbc; ones(1,size(qdbc,2))];
    ck = ak*B;
    cbck = ak*BBc;
    C(k,:) = ck;
    CBc(k,:)=cbck;
    
    %%%
    gbc = activation(CBc(1:k,:));
    g = activation(C(1:k,:));
    dg = dactivation(C(1:k,:));
    d2g = d2activation(C(1:k,:));


    % c     The stiffness matrix of 2D Helmholtz equation
    % c     a(gi,gj,mu) = (lap(gi)+k*gi, lap(gj)+k*gj) + mu*(gi, gj)_{boundary}

     GLaplace = ((dg.*(w*lapp))*(dg.*(w*lapp).*wei)') + ...                              % lap(gi) * lap(gj), part1
               ((dg.*(w*lapp))*(d2g.*((w*dxp).^2+(w*dyp).^2).*wei)') + ...              % lap(gi) * lap(gj), part2
               ((d2g.*((w*dxp).^2+(w*dyp).^2))*(dg.*(w*lapp).*wei)') + ...              % lap(gi) * lap(gj), part3
               ((d2g.*((w*dxp).^2+(w*dyp).^2))*(d2g.*((w*dxp).^2+(w*dyp).^2).*wei)');   % lap(gi) * lap(gj), part4
    
    GCross = ((dg.*(w*lapp))*(g.*wei)') + ...                   % lap(gi) * gj + gi * lap(gj), part1
             ((d2g.*((w*dxp).^2+(w*dyp).^2))*(g.*wei)') + ...   % lap(gi) * gj + gi * lap(gj), part2
             (g*(dg.*(w*lapp).*wei)') + ...                     % lap(gi) * gj + gi * lap(gj), part3
             (g*(d2g.*((w*dxp).^2+(w*dyp).^2).*wei)');          % lap(gi) * gj + gi * lap(gj), part4

    GLinear = (g*(g.*wei)');    % gi * gj, in the domain

    GBc = (gbc*(gbc.*weibc)');  % gi * gj, on boundary  

    GHelmholtz = (GLaplace + wn*GCross + wn^2*GLinear)*(h/2)*(h/2) + mu*GBc*(h/2); 
    

    % c     The load vector of 2D Helmholtz equation
    % c     F(gi,mu) = (f, lap(gi)+k*gi) + mu*(gD, gi)_{boundary}

    bLaplace = (dg.*(w*lapp))*f + (d2g.*((w*dxp).^2+(w*dyp).^2))*f;
    bLinear = g*f;
    bBc = (gbc*gD);
    bHelmholtz = (bLaplace + wn*bLinear)*(h/2)*(h/2) + mu*bBc*(h/2);


    % c     Solve and clear 
    xk = GHelmholtz \ bHelmholtz;
    W2(1:k) = xk;
    cg = xk';
    clear xk ck 
    clear bHelmholtz GHelmholtz GLaplace GCross GLinear GBc

    % c     The new projection as the k-th numerical solution. New neuron
    % c     added into the shallow network and coefficients updated for 
    % c     all neurons.
    
    uk = @(x)(cg*activation(w*Net.preLayer(x)+b)); 
    helmholtz_uk = @(x)( ...
            cg*(d2activation(w*Net.preLayer(x)+b).*((w*Net.dxpreLayer(x)).^2 + (w*Net.dypreLayer(x)).^2)) + ...
            cg*(dactivation(w*Net.preLayer(x)+b).*(w*Net.lappreLayer(x))) + ...
            cg*activation(w*Net.preLayer(x)+b) * wn ... 
            );
    
    % c     Computation of numerical errors. Errors in L_2 norm and H^1
    % c     norm will be displayed. The theoretical convergence order in 
    % c     H^m norm is  1/2 + (2(k-m)+1)/2d
    % c     where k is the power of ReLU and d is the dimension.

    l2e = (pde.residual(qdpts) - uk(qdpts)).^2;
    errl2 = sqrt(sum(l2e.*wei)*h/2);

    loss_in = (helmholtz_uk(qdpts)-auxInfo.f).^2;
    loss_bc = mu * (uk(qdbc)-auxInfo.gD).^2;
    loss_al = (1/2)*sum(loss_in.*wei)*(h/2)*(h/2) + (1/2)*sum(loss_bc.*weibc)*(h/2);

    errL2(k) = errl2;
    Loss(k) = loss_al;

    %%% outputs
    fprintf('the L2 fitting error = %e \n', errl2)
    fprintf('the PINN loss = %e \n', loss_al)
    
end


%% add four nodes using the relation:

if layer > 0
    [W2, W1, Bias] = residualCorrection(W2, W1, Bias, degree, paramPreLayer);
end

%% outputs
parameters  = [W2, W1, Bias];
uk = @(x)(W2'*activation(W1*Net.preLayer(x)+Bias)); 
helmholtz_uk = @(p)( ...
        W2'*(d2activation(W1*Net.preLayer(p)+Bias).*((W1*Net.dxpreLayer(p)).^2 + (W1*Net.dypreLayer(p)).^2)) + ...
        W2'*(dactivation(W1*Net.preLayer(p)+Bias).*(W1*Net.lappreLayer(p))) + ...
        W2'*activation(W1*Net.preLayer(p)+Bias) * wn ...
        );
if layer > 0
    errL2 = [errL2; errl2*ones(2*degree,1)];
    Loss = [Loss; loss_al*ones(2*degree,1)];
end
fprintf('the L2 fitting error = %e \n', errl2)
fprintf('the PINN loss = %e \n', loss_al)

%% testing PINN loss

%%% test dataset
% pts = 2;
% hTest = 1/100;
% [qdTest,weiTest] = rquadpts2d(rectangle,pts,100);
% [qdTestBc, weiTestBc] = rquadpts2dbc(rectangle, pts, 100);

%% the vector of neurons of the current layer
preLayer = Net.preLayer;
dxpreLayer = Net.dxpreLayer;
dypreLayer = Net.dypreLayer;
lappreLayer = Net.lappreLayer;
Net.preLayer = @(p)(activation(W1*preLayer(p)+Bias)); 
Net.dxpreLayer = @(p)(dactivation(W1*preLayer(p)+Bias).*(W1*dxpreLayer(p)));
Net.dypreLayer = @(p)(dactivation(W1*preLayer(p)+Bias).*(W1*dypreLayer(p)));
Net.lappreLayer = @(p)(d2activation(W1*preLayer(p)+Bias).*(W1*dxpreLayer(p)).^2 + ...
                       d2activation(W1*preLayer(p)+Bias).*(W1*dypreLayer(p)).^2 + ...
                       dactivation(W1*preLayer(p)+Bias).*(W1*lappreLayer(p)));

%% current approximator
approximator.self = uk;
approximator.helmholtz = helmholtz_uk;

end
