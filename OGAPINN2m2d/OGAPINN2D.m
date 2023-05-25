function [errL2, errA, loss, dnn] = OGAPINN2D(level, pde, option)

%% initialization and activation functions
uk = @(x,y)(0);
dxuk = @(x,y)(0);  % dx
dyuk = @(x,y)(0);  % dy
dxxuk = @(x,y)(0); % dxx
dyyuk = @(x,y)(0); % dyy
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
N = 400;
pts = 2;
h = [1/N,1/N];
hx = h(1); hy = h(2);
rectangle = [0,1;0,1];
[xqd,yqd,wei] = rquadpts2d(rectangle,pts,N);
gquad.hx = hx;  gquad.hy = hy;
gquad.wei = wei; gquad.xqd = xqd;  
gquad.yqd = yqd; gquad.numpts = length(xqd);
clear N hx hy xqd yqd wei

%% monte-carlo information
N = 20000;
[xqd,yqd,wei] = rmcpts2d(rectangle,N);
mcquad.xqd = xqd; mcquad.yqd = yqd;
mcquad.wei = wei; mcquad.numpts = length(xqd);
clear N xqd yqd wei

%% boundary quadrature
Nbc = 1000;
x_min = rectangle(1,1); x_max = rectangle(1,2);
y_min = rectangle(2,1); y_max = rectangle(2,2);
interval_x = rectangle(1,:);
interval_y = rectangle(2,:);
[xqd,wei_x] = rmcpts1d(interval_x,Nbc);
[yqd,wei_y] = rmcpts1d(interval_y,Nbc);
mcquadbc.xqd = [xqd,...
                xqd,...
                x_min*ones(1,Nbc),...
                x_max*ones(1,Nbc)];
mcquadbc.yqd = [y_min*ones(1,Nbc),...
                y_max*ones(1,Nbc),...
                yqd,...
                yqd];
mcquadbc.wei = [wei_x,wei_x,wei_y,wei_y];
mcquadbc.numpts = length(mcquadbc.xqd);
isboundaryx = ones(size(wei_x));
isboundaryy = ones(size(wei_y));
mcquadbc.indicatorx = [~isboundaryx,~isboundaryx,-isboundaryx,isboundaryx];
mcquadbc.indicatory = [-isboundaryy,isboundaryy,~isboundaryy,~isboundaryy];
clear Nbc xqd yqd wei_x wei_y isboundaryx isboundaryy

%% parameters initialization
nNeuron = 2^level;            % number of neurons
W1 = zeros(nNeuron,1);        % w1
W2 = zeros(nNeuron,1);        % w2
Bias = zeros(nNeuron,1);      % b
C = zeros(nNeuron,mcquad.numpts);
Cbc = zeros(nNeuron,mcquadbc.numpts);

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
    
    auxInfo.uk = uk(mcquad.xqd,mcquad.yqd);
    auxInfo.dxuk = dxuk(mcquadbc.xqd,mcquadbc.yqd);    
    auxInfo.dyuk = dyuk(mcquadbc.xqd,mcquadbc.yqd);
    auxInfo.dxxuk = dxxuk(mcquad.xqd,mcquad.yqd);
    auxInfo.dyyuk = dyyuk(mcquad.xqd,mcquad.yqd);
    auxInfo.f = pde.rhs(mcquad.xqd,mcquad.yqd);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,mcquad.numpts);
        auxInfo.dxuk = repmat(auxInfo.dxuk,1,mcquadbc.numpts);
        auxInfo.dyuk = repmat(auxInfo.dyuk,1,mcquadbc.numpts);
        auxInfo.dxxuk = repmat(auxInfo.dxxuk,1,mcquad.numpts);
        auxInfo.dyyuk = repmat(auxInfo.dyyuk,1,mcquad.numpts);
    end
 
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration.

    tic
    start_time = clock;
    square = [-2,2];
    theta = initialguess(mcquad,mcquadbc,square,auxInfo);
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
    B = [mcquad.xqd;mcquad.yqd;ones(size(mcquad.xqd))];
    Bbc = [mcquadbc.xqd;mcquadbc.yqd;ones(size(mcquadbc.xqd))];
    ck = ak*B;
    cbck = ak*Bbc;
    C(k,:) = ck;
    Cbc(k,:)=cbck;
    g = activation(C(1:k,:));
    d2g = d2activation(C(1:k,:));

    %%% part 1
    I = ones(k,1);
    G = (g*(g.*mcquad.wei)');                              % gi*gj
    Gk = G;
    clear G

    %%% part 2
    G21 = -(d2g*(g.*mcquad.wei)').*((w1.^2)*(I.^2)');      % dxx(gi)*gj
    G22 = -(d2g*(g.*mcquad.wei)').*((w2.^2)*(I.^2)');      % dyy(gi)*gj
    Gk = Gk + G21 + G22;
    clear G21 G22

    %%% part 3
    G31 = -(g*(d2g.*mcquad.wei)').*((I.^2)*(w1.^2)');      % gi*dxx(gj)
    G32 = -(g*(d2g.*mcquad.wei)').*((I.^2)*(w2.^2)');      % gi*dyy(gj)
    Gk = Gk + G31 + G32;
    clear G31 G32

    %%% part 4
    G41 = (d2g*(d2g.*mcquad.wei)').*((w1.^2)*(w1.^2)');    % dxx(gi)*dxx(gj)
    G42 = (d2g*(d2g.*mcquad.wei)').*((w1.^2)*(w2.^2)');    % dxx(gi)*gyy(gj)
    G43 = (d2g*(d2g.*mcquad.wei)').*((w2.^2)*(w1.^2)');    % dyy(gi)*dxx(gj)
    G44 = (d2g*(d2g.*mcquad.wei)').*((w2.^2)*(w2.^2)');    % dyy(gi)*dyy(gj)
    Gk = Gk + G41 + G42 + G43 + G44;
    clear G41 G42 G43 G44

    %%% part 5, boundary integration
    dg = dactivation(Cbc(1:k,:));
    dxg = dg.*mcquadbc.indicatorx;
    dyg = dg.*mcquadbc.indicatory;
    G51 = (dxg*(dxg.*mcquadbc.wei)').*(w1*w1');      % dx(gi)*dx(gj)
    G52 = (dxg*(dyg.*mcquadbc.wei)').*(w1*w2');      % dx(gi)*dy(gj)
    G53 = (dyg*(dxg.*mcquadbc.wei)').*(w2*w1');      % dy(gi)*dx(gj)
    G54 = (dyg*(dyg.*mcquadbc.wei)').*(w2*w2');      % dy(gi)*dy(gi)
    Gk = Gk + (G51 + G52 + G53 + G54);
    clear dg dxg dyg G51 G52 G53 G54

    %%% part 6, source term
    f = (auxInfo.f.*mcquad.wei)';
    bk = (-d2g)*f.*(w1.^2) + (-d2g)*f.*(w2.^2) + g*f;   % f*(-dxx(gi)-dyy(gi)+gi)
    clear g d2g f

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
    dxxuk = @(x,y)(cg*(d2activation(w1*x+w2*y+b).*w1.*w1));
    dyyuk = @(x,y)(cg*(d2activation(w1*x+w2*y+b).*w2.*w2));

    % c     Computation of numerical errors. Errors in L_2 norm and energy
    % c     norm will be displayed. The theoretical convergence order in 
    % c     H^m norm is  1/2 + (2(k-m)+1)/2d
    % c     where k is the power of ReLU and d is the dimension.

    l2e = (pde.target(gquad.xqd,gquad.yqd) - uk(gquad.xqd,gquad.yqd)).^2;
    dxe = (pde.dxtarget(gquad.xqd,gquad.yqd) - dxuk(gquad.xqd,gquad.yqd)).^2;
    dye = (pde.dytarget(gquad.xqd,gquad.yqd) - dyuk(gquad.xqd,gquad.yqd)).^2;
    errl2 = sqrt(sum(l2e.*gquad.wei)*(gquad.hx/2)*(gquad.hy/2));
    erra = sqrt(sum((l2e+dxe+dye).*gquad.wei)*(gquad.hx/2)*(gquad.hy/2)); 

    % c     Computation of the numerical loss function, The PINN's loss formulation 
    % c     can be written as
    % c     L(u) = \| -Delta(u)+u-f \|^2_{2} + \| du/dn \|^2_{2,boundary} 

    loss_in = (-dxxuk(mcquad.xqd,mcquad.yqd) -dyyuk(mcquad.xqd,mcquad.yqd) + ...
                uk(mcquad.xqd,mcquad.yqd)-auxInfo.f).^2;
    loss_bc = (dxuk(mcquadbc.xqd,mcquadbc.yqd).*mcquadbc.indicatorx + ...
               dyuk(mcquadbc.xqd,mcquadbc.yqd).*mcquadbc.indicatory).^2;
    loss_al = sum(loss_in.*mcquad.wei) + sum(loss_bc.*mcquadbc.wei);
    
    errL2(k) = errl2;
    errA(k) = erra;
    loss(k) = loss_al;
    
    %%% outputs
    fprintf('the L2 fitting error = %e \n',errl2)
    fprintf('the EN fitting error = %e \n',erra)
    fprintf('the loss function = %e \n', loss_al)
    
end
dnn = [cg', W1, W2, Bias];

end

%% end of file