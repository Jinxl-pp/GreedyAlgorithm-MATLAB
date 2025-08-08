function [errL2, Loss, parameters, approximator, Net] = ogaForLayer(Net, degree, pde, interval, quadInterval)
%% initialization and activation functions
uk = @(x)(0);  % u
d2uk = @(x)(0); % Lap(u)
mu = 1;
auxInfo.mu = mu; % penalty parameter for BC
auxInfo.deg = degree;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);

%% largest value of L2 norm of neurons from last layer
x = linspace(interval(1),interval(2),1e4);
c = Net.preLayer(x);
c = max(sqrt(sum(c.^2,1))) + 1;
square = [-c,c];

%% quadrature information
% c     Guass quadreture points in 1D: discretize the interval and apply 
% c     2pts Gauss quadrature rule on each element.
% c     Guass quadreture points in 2D: 2D tensor product of 1D 
% c     Guass quadreture points in 3D: 3D tensor product of 1D 
pts = 2;
h = 1/quadInterval;
[xqd,wei] = rquadpts1d(interval,pts,quadInterval);
numPts = length(xqd);
auxInfo.quadInterval = quadInterval;
auxInfo.f = pde.rhs(xqd);
f = (auxInfo.f.*wei)';

%% quadrature on boudary
% c     For 1D, the quadrature is just a set of two points and uniform weights
xqdbc = interval;
weibc = [1, 1];
auxInfo.gD = pde.Dbc(xqdbc);
gD = (auxInfo.gD.*weibc)';

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
p = Net.preLayer(xqd);
dp = Net.dpreLayer(xqd);
d2p = Net.d2preLayer(xqd);
pbc = Net.preLayer(xqdbc);
auxInfo.dim = inputDim;
auxInfo.p = p;
auxInfo.dp = dp;
auxInfo.d2p = d2p;
auxInfo.pbc = pbc;

%% parameters initialization
W2 = zeros(trainingSteps,1);        
W1 = zeros(trainingSteps,inputDim);
Bias = zeros(trainingSteps,1);      
C = zeros(trainingSteps,numPts);
Cbc = zeros(trainingSteps,2);

%% Orthogonal greedy algorithm
timeList = zeros(trainingSteps,1);
errL2 = zeros(trainingSteps,1);
Loss = zeros(trainingSteps,1);
for k = 1:trainingSteps

    % c     The subproblem is key to the greedy algorithm, where the inner
    % c     products |(u,g) - (f,g)| should be maximized.
    % c     Part of the inner products can be computed in advance.

    auxInfo.uk = uk(xqd); 
    auxInfo.d2uk = d2uk(xqd);
    auxInfo.ukbc = uk(xqdbc);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,numPts);
        auxInfo.d2uk = repmat(auxInfo.d2uk,1,numPts);
        auxInfo.ukbc = repmat(auxInfo.ukbc,1,2);
    end
    
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration. Since |w|=1, only 
    % c     consider w = +-1.

    tic
    start_time = clock;
    theta = optimalBasis(xqd,wei,xqdbc,weibc,square,auxInfo,h);
    fprintf('------------ the N = %d-th neuron ------------ \n', k);
    toc
    end_time=clock;
    timeList(k) = etime(end_time, start_time);
    
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
    B = [p; ones(size(xqd))];
    Bbc = [pbc; ones(size(xqdbc))];
    ck = ak*B;
    cbck = ak*Bbc;
    C(k,:) = ck;
    Cbc(k,:)=cbck;
    
    %%%
    gbc = activation(Cbc(1:k,:));
    dg = dactivation(C(1:k,:));
    d2g = d2activation(C(1:k,:));
    
    %%%
    d2G = ((d2g.*(w*dp).^2)*(d2g.*(w*dp).^2.*wei)')*(h/2) ... % dxx(gi) * dxx(gj), part1
        + ((d2g.*(w*dp).^2)*(dg.*(w*d2p).*wei)')*(h/2) ...    % dxx(gi) * dxx(gj), part2
        + ((dg.*(w*d2p))*(d2g.*(w*dp).^2.*wei)')*(h/2) ...    % dxx(gi) * dxx(gj), part3
        + ((dg.*(w*d2p))*(dg.*(w*d2p).*wei)')*(h/2);          % dxx(gi) * dxx(gj), part4
    
    Gbc = mu * (gbc*(gbc.*weibc)');    % gi * gj, on boundary
    Gk = d2G + Gbc;
            
    %%%
    bk = (-d2g.*(w*dp).^2)*f*(h/2) + (-dg.*(w*d2p))*f*(h/2) ...    % f * (Lap(gi)), part1
       + mu * (gbc*gD) ;
    %%%
    xk = Gk \ bk;
    W2(1:k) = xk;
    cg = xk';
    clear ck bk xk G Gk Gd

    % c     The new projection as the k-th numerical solution. New neuron
    % c     added into the shallow network and coefficients updated for 
    % c     all neurons.
    
    uk = @(x)(cg*activation(w*Net.preLayer(x)+b)); 
    d2uk = @(x)( ...
            cg*(d2activation(w*Net.preLayer(x)+b).*(w*Net.dpreLayer(x)).^2) + ...
            cg*(dactivation(w*Net.preLayer(x)+b).*(w*Net.d2preLayer(x))) ...
            );
    
    % c     Computation of numerical errors. Errors in L_2 norm and H^1
    % c     norm will be displayed. The theoretical convergence order in 
    % c     H^m norm is  1/2 + (2(k-m)+1)/2d
    % c     where k is the power of ReLU and d is the dimension.

    l2e = (pde.residual(xqd) - uk(xqd)).^2;
    errl2 = sqrt(sum(l2e.*wei)*h/2);

    loss_in = (-d2uk(xqd)-auxInfo.f).^2;
    loss_bc = mu * (uk(xqdbc)-auxInfo.gD).^2;
    loss_al = sum(loss_in.*wei)*(h/2) + sum(loss_bc);

    errL2(k) = errl2;
    Loss(k) = loss_al;

    %     if mod(k,10)==0
    %         x = linspace(interval(1),interval(2),1e5);
    %         y = uk(x);%-pde.target(x);
    %         plot(x,y, 'r-')
    %     end

    %%% outputs
    fprintf('the L2 fitting error = %e \n', errl2)
    fprintf('the PINN loss = %e \n', loss_al)
    
end


%% add four nodes using the relation:
% x = (1/2)*ReLU(sqrt(2)/2*(x+1)) + (1/2)*ReLU(-sqrt(2)/2*(x+1)) 
%   - (1/2)*ReLU(sqrt(2)/2*(x-1)) - (1/2)*ReLU(-sqrt(2)/2*(x-1))

if layer > 0
%     W2 = [W2; 1/2; 1/2; -1/2; -1/2];
%     W1 = [W1; sqrt(2)/2*paramPreLayer(:,1)'; -sqrt(2)/2*paramPreLayer(:,1)';
%               sqrt(2)/2*paramPreLayer(:,1)'; -sqrt(2)/2*paramPreLayer(:,1)'];
%     Bias = [Bias; sqrt(2)/2; -sqrt(2)/2;
%                   -sqrt(2)/2; sqrt(2)/2];
    [W2, W1, Bias] = residualCorrection(W2, W1, Bias, degree, paramPreLayer);
end

%% outputs
parameters  = [W2, W1, Bias];
uk = @(x)(W2'*activation(W1*Net.preLayer(x)+Bias)); 
d2uk = @(x)( ...
        W2'*(d2activation(W1*Net.preLayer(x)+Bias).*(W1*Net.dpreLayer(x)).^2) + ...
        W2'*(dactivation(W1*Net.preLayer(x)+Bias).*(W1*Net.d2preLayer(x))) ...
        );
if layer > 0
    errL2 = [errL2; errl2*ones(2*degree,1)];
    Loss = [Loss; loss_al*ones(2*degree,1)];
end
fprintf('the L2 fitting error = %e \n', errl2)
fprintf('the PINN loss = %e \n', loss_al)

%% the vector of neurons of the current layer
preLayer = Net.preLayer;
dpreLayer = Net.dpreLayer;
d2preLayer = Net.d2preLayer;
Net.preLayer = @(x)(activation(W1*preLayer(x)+Bias)); 
Net.dpreLayer = @(x)(dactivation(W1*preLayer(x)+Bias).*(W1*dpreLayer(x)));
Net.d2preLayer = @(x)(d2activation(W1*preLayer(x)+Bias).*(W1*dpreLayer(x)).^2 + ...
                      dactivation(W1*preLayer(x)+Bias).*(W1*d2preLayer(x)));

%% current approximator
approximator.self = uk;
approximator.lap = d2uk;

end
