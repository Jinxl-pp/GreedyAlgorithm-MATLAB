%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c
% c    2021 Wenrui Hao, Xianlin Jin, Jonathan W Seigel, Jinchao Xu
% c
% c    This file tests the convergence rate of orthogonal greedy
% c    algorithm for solving a one-dimensional second order PDE
% c                     (-жд)u + u = f, x \in (-1,1)
% c    with zero Neumann-type boundary condition
% c                         du/dn = 0,
% c    where dn stands for the differentiation along the normal vector.
% c    The dictionary we use to build the shallow neural network is 
% c    chosen as
% c                 {\ReLU^{k}(wx+b); |w|=1, b \in [-2,2]}.
% c    The theoretical convergence order w.r.t the number of
% c    neurons are:
% c           4 in L2 norm and 3 in H1 norm, resp(k=3).
% c           3 in L2 norm and 2 in H1 norm, resp(k=2).
% c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
% clc;
% parpool('local',8)
addpath('../tool','../data')
% pde = datacos1d;
pde = datapeak1d;
profile on

%% initialization and activation functions
uk = @(x)(0);  % u
duk = @(x)(0); % dx(u)
degree = 2;    % 3
auxInfo.deg = degree;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);

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
numpts = length(xqd);

%% parameters initialization
nNeuron = 2^6;                % number of neurons
W = zeros(nNeuron,1);         % w
Bias = zeros(nNeuron,1);      % b
C = zeros(nNeuron,numpts);

%% training options
% c     Three training options: 
% c     1, gradient descent method, 
% c     2, finite difference Newton's method with explicit Euler scheme,
% c     3, finite difference Newton's method with central difference scheme.

option = 1;
% option = 2;
% option = 3;

%% Orthogonal greedy algorithm
time_list = zeros(nNeuron,1);
warning_count = 0;
errL2 = zeros(nNeuron,1);
errH1 = zeros(nNeuron,1);
for k = 1 : nNeuron
    % c     The subproblem is key to the greedy algorithm, where the inner
    % c     products |(\nabla u_k, \nabla g) + (u,g) - (f,g)| should be
    % c     maximized. Part of the inner products can be computed in advance.

    auxInfo.uk = uk(xqd); 
    auxInfo.duk = duk(xqd);
    auxInfo.f = pde.rhs(xqd);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,numpts);
        auxInfo.duk = repmat(auxInfo.duk,1,numpts);
    end
    
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration. Since |w|=1, only 
    % c     consider w = +-1.

    tic
    start_time = clock;
    square = [-2,2];
    theta = initialguess(xqd,wei,square,auxInfo,h);
    fprintf('------------ the N = %d-th neuron ------------ \n', k);
    toc
    end_time=clock;
    time_list(k) = etime(end_time, start_time);
    
    % c     A few steps of training are needed to reach a local minimum 
    % c     near  the initial guess. Gradient descent, coordinate descent,
    % c     Newton's method, etc., can be applied to this problem.

    [E0,dE0] = gradientdescent(theta,xqd,wei,auxInfo,h);

    switch option
        %%% gradient descent
        case 1 
            eta = 1e-6;
            num_epoc = 0;
            for i = 1:num_epoc
                [E,dE] = gradientdescent(theta,xqd,wei,auxInfo,h);
                grad=norm(dE,inf);
                theta=theta-eta*dE;
                if mod(i,50)==0
                    fprintf('iter=%d,loss=%e,grad=%e\n',i,E,grad)
                end
            end

        %%% finite difference Newton 1
        case 2
            eta = 1e-4;
            epsilon = 1e-6;
            num_epoc = 0;
            for i = 1:num_epoc
                [E,dE] = gradientdescent(theta,xqd,wei,auxInfo,h);
                dh = zeros(2,1);
                dh(2) = epsilon;%*abs(theta(2))*sign(theta(2));
                [Eb,dEb] = gradientdescent(theta+dh,xqd,wei,auxInfo,h);
                HE = dh(2)/(dEb(2)-dE(2));
                grad=norm(dE,inf);
                theta(2)=theta(2)-eta*HE*dE(2);
                if mod(i,50)==0
                    fprintf('iter=%d,loss=%1.10e,grad=%e\n',i,E,grad)
                end
            end

        %%% finite difference Newton 2
        case 3
            eta = 1e-6;
            epsilon = 1e-8;
            num_epoc = 0;
            for i = 1:num_epoc
                HE = zeros(2,2);
                [E,dE] = gradientdescent(theta,xqd,wei,auxInfo,h);
                for j = 1:2
                    dh = zeros(2,1);
                    dh(j) = epsilon*theta(j);
                    [E1j,dE1j] = gradientdescent(theta+dh,xqd,wei,auxInfo,h);
                    [E2j,dE2j] = gradientdescent(theta-dh,xqd,wei,auxInfo,h);
                    HE(:,j) = (dE1j-dE2j)/dh(j)/2;
                end
                HE = (HE+HE')/2;
                grad=norm(dE,inf);
                theta=theta-eta*HE\dE;
                if mod(i,20)==0
                    fprintf('iter=%d,loss=%e,grad=%e\n',i,E,grad)
                end
            end
    end

    [E1,dE1] = gradientdescent(theta,xqd,wei,auxInfo,h);
    if (E1>E0) || (norm(dE1,inf)>norm(dE0,inf))
        warning('newton_s method failed!')
        warning_count = warning_count + 1;
    end

    W(k) = theta(1);
    Bias(k) = theta(2);
    w = W(1:k);
    b = Bias(1:k);
    
    % c     Computation of the orthogonal projection. The Gram matrix 
    % c     G(i,j) = (\nabla g_i, \nabla g_j) + (g_i, g_j)
    % c     and the RHS vector 
    % c     b(j) = (f, g_j).

    ak = [W(k),Bias(k)];
    B = [xqd;ones(size(xqd))];
    ck = ak*B;
    C(k,:) = ck;
    g = activation(C(1:k,:));
    dg = dactivation(C(1:k,:));
    %%%
    G = (g*(g.*wei)')*(h/2);                  % gi*gj
    Gd = (dg*(dg.*wei)').*(w*w')*(h/2);       % dx(gi)*dx(gj)
    Gk = Gd + G;                  
    %%%
    f = (auxInfo.f.*wei)';
    bk = g*f*(h/2);                           % f*gi
    %%%
    xk = Gk \ bk;
    cg = xk';
    clear ck bk xk G Gk Gd

    % c     The new projection as the k-th numerical solution. New neuron
    % c     added into the shallow network and coefficients updated for
    % c     all neurons.
    
    uk = @(x)(cg*activation(w*x+b)); 
    duk = @(x)(cg*(dactivation(w*x+b).*w));
    
    % c     Computation of numerical errors. Errors in L_2 norm and H^1
    % c     norm will be displayed. The theoretical convergence order in 
    % c     H^m norm is  1/2 + (2(k-m)+1)/2d
    % c     where k is the power of ReLU and d is the dimension.

    l2e = (pde.target(xqd) - uk(xqd)).^2;
    h1e = (pde.dtarget(xqd) - duk(xqd)).^2;
    errl2 = sqrt(sum(l2e.*wei)*h/2);
    errh1 = sqrt(sum((l2e + h1e).*wei)*h/2);
    errL2(k) = errl2;
    errH1(k) = errh1;
    
    %%% outputs
    fprintf('the L2 fitting error = %e \n', errl2)
    fprintf('the H1 fitting error = %e \n', errh1)
    
end

warning_count
profile viewer

%% plot the approximation
figure(1)
x = -1:0.001:1;
plot(x,uk(x),'r-')
hold on
plot(x,pde.target(x),'b-')
hold on
plot((-b./w)',uk((-b./w)'),'ro')
axis([-1,1,-1.1,1.1])
legend('approximation','target','grid points')

%% display error and convergence order
ord = 1:log2(nNeuron);
ord = (2.^ord)';
Nmesh = ord;
errh1 = errH1(ord);
errl2 = errL2(ord);
orderh1 = zeros(size(ord));
orderl2 = zeros(size(ord));
orderh1(2:end) = log2(errh1(1:end-1)./errh1(2:end));
orderl2(2:end) = log2(errl2(1:end-1)./errl2(2:end));

err = struct('N',Nmesh, ...
             'H1',errh1, 'orderH1', orderh1, ...
             'L2',errl2, 'orderL2', orderl2);
disp('Table: orthogonal greedy algorithm for solving PDE')
colname = {'#N_mesh','||u-u_h||','order', ...
                     '||Du-Du_h||','order'};
disptable(colname,err.N,[],err.L2,'%0.6e',err.orderL2,'%0.2f', ...
                           err.H1,'%0.6e',err.orderH1,'%0.2f');
                  
%% display the errors in latex tabular
J = length(Nmesh);
fprintf("$N$ & $\\|u-u_N\\|_{L_2}$ & order & " + ...
        "$\\|u-u_N\\|_{H^1}$ & order \n");
for j = 1 : J
    fprintf('%d & %e & %0.2f & %e & %0.2f \\\\ \\hline \n',...
            err.N(j),err.L2(j),err.orderL2(j),...
            err.H1(j),err.orderH1(j));
end
%% end of file