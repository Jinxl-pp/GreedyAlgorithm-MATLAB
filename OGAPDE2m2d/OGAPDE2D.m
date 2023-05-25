function [errL2, errA, dnn] = OGAPDE2D(level, pde, option)

%% initialization and activation functions
uk = @(x,y)(0);
dxuk = @(x,y)(0);  % dx
dyuk = @(x,y)(0);  % dy
degree = 3;
auxInfo.deg = degree;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
 
%% quadrature information
% c     Guass quadreture points in 1D: discretize the interval and apply 
% c     2pts Gauss quadrature rule on each element.
% c     Guass quadreture points in 2D: 2D tensor product of 1D 
% c     Guass quadreture points in 3D: 3D tensor product of 1D 
N = 50;
pts = 2;
h = [1/N,1/N];
hx = h(1); hy = h(2);
rectangle = [0,1;0,1];
[xqd,yqd,wei] = rquadpts2d(rectangle,pts,N);
gquad.hx = hx;  gquad.hy = hy;
gquad.area = (hx/2)*(hy/2);
gquad.wei = wei; gquad.xqd = xqd;  
gquad.yqd = yqd; gquad.numpts = length(xqd);
clear N hx hy xqd yqd wei

%% parameters initialization
nNeuron = 2^level;            % number of neurons
W1 = zeros(nNeuron,1);        % w1
W2 = zeros(nNeuron,1);        % w2
Bias = zeros(nNeuron,1);      % b
C = zeros(nNeuron,gquad.numpts);

%% training options
% c     Three training options: 
% c     1, gradient descent method, 
% c     2, finite difference Newton's method with explicit Euler scheme,
% c     3, finite difference Newton's method with central difference scheme.

% option = 1;
% option = 2;
% option = 3;
 
%% Orthogonal greedy algorithm
time_list = zeros(nNeuron,1);
errL2 = zeros(nNeuron,1);
errA = zeros(nNeuron,1);
loss = zeros(nNeuron,1);
for k = 1:nNeuron
    % c     The subproblem is key to the greedy algorithm, where the inner
    % c     products |(\nabla u_k, \nabla g) + (u,g) - (f,g)| should be
    % c     maximized. Part of the inner products can be computed in advance.
    
    auxInfo.uk = uk(gquad.xqd,gquad.yqd);
    auxInfo.dxuk = dxuk(gquad.xqd,gquad.yqd);    
    auxInfo.dyuk = dyuk(gquad.xqd,gquad.yqd);
    auxInfo.f = pde.rhs(gquad.xqd,gquad.yqd);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,gquad.numpts);
        auxInfo.dxuk = repmat(auxInfo.dxuk,1,gquad.numpts);
        auxInfo.dyuk = repmat(auxInfo.dyuk,1,gquad.numpts);
    end
 
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration.

    tic
    start_time = clock;
    square = [-2,2];
    theta = initialguess(gquad,square,auxInfo);
    fprintf('----------- the N = %d-th element ----------- \n', k);    
    end_time=clock;
    time_list(k) = etime(end_time, start_time);
    toc
    
    % c     A few steps of training are needed to reach a local minimum 
    % c     near  the initial guess. Gradient descent, coordinate descent,
    % c     Newton's method, etc., can be applied to this problem.

    switch option
        %%% gradient descent
        case 1 
            eta = 1e-6;
            num_epoc = 0;
            for i = 1:num_epoc
                [E,dE] = gradientdescent(theta,xqd,yqd,wei,auxInfo,h);
                grad=norm(dE,inf);
                theta=theta-eta*dE;
                if mod(i,50)==0
                    fprintf('iter=%d,loss=%e,grad=%e\n',i,E,grad)
                end
            end

        %%% finite difference Newton 1
        case 2
            eta = 1e-6;
            epsilon = 1e-8;
            num_epoc = 0;
            for i = 1:num_epoc
                HE = zeros(2,2);
                [E,dE] = gradientdescent(theta,xqd,yqd,wei,auxInfo,h);
                for j = 1:2
                    dh = zeros(2,1);
                    dh(j) = epsilon*abs(theta(j))*sign(theta(j));
                    [Ej,dEj] = gradientdescent(theta+dh,xqd,yqd,wei,auxInfo,h);
                    HE(:,j) = (dEj-dE)/dh(j);
                end
                HE = (HE+HE')/2;
                grad=norm(dE,inf);
                theta=theta-eta*pinv(HE)*dE;
            end

        %%% finite difference Newton 2
        case 3
            eta = 1e-3;
            epsilon = 1e-8;
            num_epoc = 0;
            for i = 1:num_epoc
                HE = zeros(2,2);
                for j = 1:2
                    dh = zeros(2,1);
                    dh(j) = epsilon*theta(j);
                    [E1j,dE1j] = gradientdescent(theta+dh,xqd,yqd,wei,auxInfo,h);
                    [E2j,dE2j] = gradientdescent(theta-dh,xqd,yqd,wei,auxInfo,h);
                    HE(:,j) = (dE1j-dE2j)/dh(j)/2;
                end
                HE = (HE+HE')/2;
                grad=norm(dE,inf);
                theta=theta-eta*HE\dE;
            end
    end
    
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
    B = [gquad.xqd;gquad.yqd;ones(size(gquad.xqd))];
    ck = ak*B;
    C(k,:) = ck;
    g = activation(C(1:k,:));
    dg = dactivation(C(1:k,:));

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
    dxe = (pde.dxtarget(gquad.xqd,gquad.yqd) - dxuk(gquad.xqd,gquad.yqd)).^2;
    dye = (pde.dytarget(gquad.xqd,gquad.yqd) - dyuk(gquad.xqd,gquad.yqd)).^2;
    errl2 = sqrt(sum(l2e.*gquad.wei)*(gquad.area));
    erra = sqrt(sum((l2e+dxe+dye).*gquad.wei)*(gquad.area)); 

    errL2(k) = errl2;
    errA(k) = erra;
  
    %%% outputs
    fprintf('the L2 fitting error = %e \n',errl2)
    fprintf('the EN fitting error = %e \n',erra)
    
end
dnn = [cg', W1, W2, Bias];

end

%% end of file