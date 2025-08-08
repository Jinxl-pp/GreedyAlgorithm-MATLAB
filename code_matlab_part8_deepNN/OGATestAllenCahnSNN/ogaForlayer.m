function [errL2, Loss, SNN] = ogaForlayer(level, pde, degree, OmegaT, N)
%% Solve the Allen-Cahn's Equation Using OGA with SNN.
% c     level: number of total iterations is 2^level
% c     pde: information of exact/reference solution
% c     degree: the smoothness of activation function
% c     OmegaT: physical-time domain
% c     quadNum: number of quad-points to discretize OmegaT.

%% initialization and activation functions
uk = @(x,t)(0);
dxuk = @(x,t)(0);  % dx
dtuk = @(x,t)(0);  % dt
dxxuk = @(x,t)(0); % dxx
auxInfo.deg = degree;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);

%% quadrature information
% c     Guass quadreture points in 1D: discretize the interval and apply 
% c     2pts Gauss quadrature rule on each element.
% c     Guass quadreture points in 2D: 2D tensor product of 1D 
% c     Guass quadreture points in 3D: 3D tensor product of 1D 
pts = 2;
h = [1/N,1/N];
hx = h(1); hy = h(2);
[xqd,tqd,wei] = rquadpts2d(OmegaT,pts,N);
gquad.hx = hx;  gquad.hy = hy;
gquad.area = (hx/2)*(hy/2);
gquad.wei = wei; gquad.xqd = xqd;  
gquad.tqd = tqd; gquad.numpts = length(xqd);
clear N hx hy xqd tqd wei

%% parameters initialization
nNeuron = 2^level;            % number of neurons
W1 = zeros(nNeuron,1);        % w1
W2 = zeros(nNeuron,1);        % w2
Bias = zeros(nNeuron,1);      % b
C = zeros(nNeuron,gquad.numpts);

%% Orthogonal greedy algorithm
errL2 = zeros(nNeuron,1);
Loss = zeros(nNeuron,1);
for k = 1:nNeuron
    % c     The subproblem is key to the greedy algorithm, where the 
    % c     functional derivative |(\nabla J(uk), g)| should be
    % c     maximized. Part of the inner products can be computed in advance.
    
    auxInfo.uk = uk(gquad.xqd,gquad.tqd);
    auxInfo.dxuk = dxuk(gquad.xqd,gquad.tqd);    
    auxInfo.dtuk = dtuk(gquad.xqd,gquad.tqd);
    auxInfo.dxxuk = dxxuk(gquad.xqd,gquad.tqd);
    auxInfo.f = pde.rhs(gquad.xqd,gquad.tqd);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,gquad.numpts);
        auxInfo.dxuk = repmat(auxInfo.dxuk,1,gquad.numpts);
        auxInfo.dtuk = repmat(auxInfo.dtuk,1,gquad.numpts);
        auxInfo.dxxuk = repmat(auxInfo.dxxuk,1,gquad.numpts);
    end
 
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration.

    tic
    square = [-2,2];
    theta = oprimalbasis(gquad,square,auxInfo);
    fprintf('----------- the N = %d-th element ----------- \n', k);    
    toc
    
    % c     Since |w|=|(w1,w2)|=1, a polar coordnitate is used to
    % c     simplify the calculation.

    W1(k) = cos(theta(1));
    W2(k) = sin(theta(1));
    Bias(k) = theta(2);
    w1 = W1(1:k);
    w2 = W2(1:k);
    b = Bias(1:k);
    
    % c     Computation of the orthogonal projection. The Gram matrix 
    % c     G(i,j) = (\nabla^2 g_i, \nabla^2 g_j) + (g_i, g_j)
    % c     and the RHS vector 
    % c     b(j) = (f, g_j).

    ak = [W1(k),W2(k),Bias(k)];
    B = [gquad.xqd;gquad.tqd;ones(size(gquad.xqd))];
    ck = ak*B;
    C(k,:) = ck;
    g = activation(C(1:k,:));
    dg = dactivation(C(1:k,:));
    d2g = d2activation(C(1:k,:));

    %%%%%%%%%%%% 2025.1.8 from here. Ps, optimal-basis

    %%% part 1
    I = ones(k,1);
    G = (g*(g.*gquad.wei)') * (gquad.area);                % gi*gj
    Gk = G;
    clear G

    %%% part 2
    G1 = (dg*(dg.*gquad.wei)').*(w1*w1') * (gquad.area);    % dxx(gi)*dxx(gj)
    G2 = (dg*(dg.*gquad.wei)').*(w2*w2') * (gquad.area);    % dyy(gi)*dyy(gj)
    Gk = Gk + G1 + G2;
    clear G1 G2

    %%% part 3, source term
    f = (auxInfo.f.*gquad.wei)';
    bk = g*f * (gquad.area);                                % f*gi
    clear f

    %%% orthogonal projection
    xk = Gk \ bk;
    cg = xk';
    clear ck cbck bk xk Gk

    % c     The new projection as the k-th numerical solution. New neuron
    % c     added into the shallow network and coefficients updated for
    % c     all neurons.

    uk = @(x,y)(cg*activation(w1*x+w2*y+b));
    dxuk = @(x,y)(cg*(dactivation(w1*x+w2*y+b).*w1));
    dyuk = @(x,y)(cg*(dactivation(w1*x+w2*y+b).*w2));

    % c     Computation of numerical errors. Errors in L_2 norm and energy
    % c     norm will be displayed. The theoretical convergence order in 
    % c     H^m norm is  1/2 + (2(k-m)+1)/2d
    % c     where k is the power of ReLU and d is the dimension.

    l2e = (pde.target(gquad.xqd,gquad.yqd) - uk(gquad.xqd,gquad.yqd)).^2;
    errl2 = sqrt(sum(l2e.*gquad.wei)*(gquad.area));
    errL2(k) = errl2;
  
    %%% outputs
    fprintf('the L2 fitting error = %e \n',errl2)
    fprintf('the EN fitting error = %e \n',erra)
    
end
SNN = [cg', W1, W2, Bias];

end


