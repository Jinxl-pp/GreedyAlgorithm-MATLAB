function [errL2, errA, loss, dnn] = OGAPINN1D(level, pde, option)

%% initialization and activation functions
uk = @(x)(0);
duk = @(x)(0);  % dx
d2uk = @(x)(0); % dxx
degree = 3;
auxInfo.deg = degree;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);
 
%% quadrature information
% c     Guass quadreture points in 1D: discretize the interval and apply 
% c     2pts Gauss quadrature rule on each element.
% c     Guass quadreture points in 2D: 2D tensor product of 1D 
% c     Guass quadreture points in 3D: 3D tensor product of 1D 

N = 1000;
pts = 2;
h = 1/N;
interval = [-1,1];
[xqd,wei] = rquadpts1d(interval,pts,N);
gq.h = h;
gq.xqd = xqd;
gq.wei = wei;
gq.numpts = length(xqd);
clear N pts h interval

%% monte-carlo information
N = 10000;
h = 2;
interval = [-1,1];
[xqd,wei] = rmcpts1d(interval,N);
quad.xqd = xqd;
quad.wei = wei;
quad.numpts = length(xqd);

%% boundary quadrature
xqdbc = [-1,1];
weibc = [1,1];
quadbc.xqd = xqdbc;
quadbc.wei = weibc;
quadbc.numpts = 2;

%% parameters initialization
nNeuron = 2^level;                % number of neurons
W = zeros(nNeuron,1);             % w
Bias = zeros(nNeuron,1);          % b
C = zeros(nNeuron,quad.numpts);
Cbc = zeros(nNeuron,quadbc.numpts);

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
    % c     products |(\nabla^2 u_k, \nabla^2 g) + (u,g) - (f,g)| should be
    % c     maximized. Part of the inner products can be computed in advance.
    
    auxInfo.uk = uk(quad.xqd); 
    auxInfo.duk = duk(quadbc.xqd);
    auxInfo.d2uk = d2uk(quad.xqd);
    auxInfo.f = pde.rhs(quad.xqd);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,quad.numpts);
        auxInfo.duk = repmat(auxInfo.duk,1,quadbc.numpts);
        auxInfo.d2uk = repmat(auxInfo.d2uk,1,quad.numpts);
    end
 
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration.

    tic
    start_time = clock;
    square = [-2,2];
    theta = initialguess(quad,quadbc,square,auxInfo,h);
    fprintf('----------- the N = %d-th element ----------- \n', k);
    toc
    end_time=clock;
    time_list(k) = etime(end_time, start_time);
    
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
            num_epoc = 5;
            for i = 1:num_epoc
                HE = zeros(2,2);
                [~,dE] = gradientdescent(theta,xqd,yqd,wei,auxInfo,h);
                for j = 1:2
                    dh = zeros(2,1);
                    dh(j) = epsilon*abs(theta(j))*sign(theta(j));
                    [~,dEj] = gradientdescent(theta+dh,xqd,yqd,wei,auxInfo,h);
                    HE(:,j) = (dEj-dE)/dh(j);
                end
                HE = (HE+HE')/2;
                grad=norm(dE,inf);
                if mod(i,2)==0
                    fprintf('iter=%d,loss=%e,grad=%e\n',i,E,grad)
                end
                theta=theta-eta*pinv(HE)*dE;
            end

        %%% finite difference Newton 2
        case 3
            eta = 1e-3;
            epsilon = 1e-8;
            num_epoc = 20;
            for i = 1:num_epoc
                HE = zeros(2,2);
                for j = 1:2
                    dh = zeros(2,1);
                    dh(j) = epsilon*theta(j);
                    [~,dE1j] = gradientdescent(theta+dh,xqd,yqd,wei,auxInfo,h);
                    [~,dE2j] = gradientdescent(theta-dh,xqd,yqd,wei,auxInfo,h);
                    HE(:,j) = (dE1j-dE2j)/dh(j)/2;
                end
                HE = (HE+HE')/2;
                grad=norm(dE,inf);
                if mod(i,2)==0
                    fprintf('iter=%d,loss=%e,grad=%e\n',i,E,grad)
                end
                theta=theta-eta*HE\dE;
            end
    end
    
    % c     Since |w|=|(w1,w2)|=1, a polar coordnitate is used to
    % c     simplify the calculation.

    W(k) = theta(1);
    Bias(k) = theta(2);
    w = W(1:k);
    b = Bias(1:k);
    
    % c     Computation of the orthogonal projection. The Gram matrix 
    % c     G(i,j) = (\nabla^2 g_i, \nabla^2 g_j) + (g_i, g_j)
    % c     and the RHS vector 
    % c     b(j) = (f, g_j).

    ak = [W(k),Bias(k)];
    B = [xqd;ones(size(xqd))];
    Bbc = [quadbc.xqd;ones(size(quadbc.xqd))];
    ck = ak*B;
    cbck = ak*Bbc;
    C(k,:) = ck;
    Cbc(k,:)=cbck;
    g = activation(C(1:k,:));
    dg = dactivation(Cbc(1:k,:));
    d2g = d2activation(C(1:k,:));
    
    %%%
    I = ones(k,1);
    G1 = (g*(g.*wei)')*(h/2);                           % gi * gj
    G2 = (g*(d2g.*wei)').*((I.^2)*(w.^2)')*(h/2);       % gi * dxx(gj)
    G3 = (d2g*(g.*wei)').*((w.^2)*(I.^2)')*(h/2);       % dxx(gi) * gj
    G4 = (d2g*(d2g.*wei)').*((w.^2)*(w.^2)')*(h/2);     % dxx(gi) * dxx(gj)
    
    %%%
    G5 = (dg*(dg.*quadbc.wei)').*(w*w');                % dx(gi) * dx(gj) on boundary
    Gk = G1 - G2 - G3 + G4 + G5;                        % (1,-1,-1,1,1)
%     clear G1 G2 G3 G4 G5  

    %%%
    f = (auxInfo.f.*wei)';
    bk = (-d2g)*f.*(w.^2)*(h/2) + g*f*(h/2);  

    %%%
    xk = Gk \ bk;
    cg = xk';
    clear ck bk xk Gk  

    % c     The new projection as the k-th numerical solution. New neuron
    % c     added into the shallow network and coefficients updated for
    % c     all neurons.

    uk = @(x)(cg*activation(w*x+b)); 
    duk = @(x)(cg*(dactivation(w*x+b).*w));
    d2uk = @(x)(cg*(d2activation(w*x+b).*w.*w));

    % c     Computation of numerical errors. Errors in L_2 norm and energy
    % c     norm will be displayed. The theoretical convergence order in 
    % c     H^m norm is  1/2 + (2(k-m)+1)/2d
    % c     where k is the power of ReLU and d is the dimension.

    l2e = (pde.target(gq.xqd) - uk(gq.xqd)).^2;
    h1e = (pde.dtarget(gq.xqd) - duk(gq.xqd)).^2;
    errl2 = sqrt(sum(l2e.*gq.wei)*gq.h/2);
    erra = sqrt(sum((l2e + h1e).*gq.wei)*gq.h/2);

    % c     Computation of the loss function. The PINN's loss formulation 
    % c     can be written as
    % c     L(u) = \| -Delta(u)+u-f \|^2 + \| du/dn \|^2_{boundary} 

    loss_in = (-d2uk(xqd)+uk(xqd)-auxInfo.f).^2;
    loss_bc = (duk(quadbc.xqd)).^2;
    loss_al = sum(loss_in.*wei)*(h/2) + sum(loss_bc.*quadbc.wei);

    %%% error lists
    errL2(k) = errl2;
    errA(k) = erra;
    loss(k) = loss_al;

    %%% outputs
    fprintf('the L2 fitting error = %e \n', errl2)
    fprintf('the EN fitting error = %e \n', erra)
    fprintf('the loss function = %e \n', loss_al)
    
end
dnn = [cg', W, Bias];

end % end of file