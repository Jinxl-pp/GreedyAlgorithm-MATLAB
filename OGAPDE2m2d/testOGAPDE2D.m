%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c
% c    2022 Xianlin Jin
% c
% c    This file tests the convergence rate of orthogonal greedy
% c    algorithm for solving a two-dimensional elliptic PDE
% c                     (-Δ)u + u = f, x \in (-1,1)^2
% c    with zero Neumann-type boundary condition
% c                         d^2u/d^2n = 0,
% c           d/dn(Δu + d^2u/d^2s) - d/ds(ks*(du/ds)) = 0,
% c    where dn, ds stand for the differentiation along the normal and
% c    tangent vector repectively, and ks is the curvature of boundary.
% c    The dictionary we use to build the shallow neural network is 
% c    chosen as
% c                 {\ReLU^{k}(wx+b); |w|=1, b \in [-2,2]}.
% c    The theoretical convergence order w.r.t the number of
% c    neurons are 9/4 in L2 norm and 5/4 in H1 norm, resp(k=3).
% c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
% clc;
% parpool('local',8)
addpath('../tool','../data')

%% OGA iteration
level = 6;
option = 1;
pde = datacos2d;
[errL2, errA, ~] = OGAPDE2D(level, pde, option);
 
%% display error and convergence order
ord = 2.^(1:level)';
Nmesh = ord;
erra = errA(ord);
errl2 = errL2(ord);
ordera = zeros(size(ord));
orderl2 = zeros(size(ord));
ordera(2:end) = log2(erra(1:end-1)./erra(2:end));
orderl2(2:end) = log2(errl2(1:end-1)./errl2(2:end));

err = struct('N', Nmesh, ...
             'A', erra, 'orderA', ordera, ...
             'L2', errl2, 'orderL2', orderl2);
disp('Table: orthogonal greedy algorithm for training PINNs')
colname = {'#N','||u-u_h||','order', ...
                '||u-u_h||_a','order'};
disptable(colname,err.N,[], err.L2,'%0.2e',err.orderL2,'%0.2f', ...
                           err.A,'%0.2e',err.orderA,'%0.2f');

%% display the errors in latex tabular form
J = length(Nmesh);
fprintf("$N$ & $ PINN-loss$ & order &" +...
        "$\\|u-u_N\\|_{L_2}$ & order & " + ...
        "$\\|u-u_N\\|_{a}$ & order \n");
for j = 1 : J
    fprintf('%d & %0.2e & %0.2f & %0.2e & %0.2f \\\\ \\hline \n',...
            err.N(j),...
            err.L2(j),err.orderL2(j),...
            err.A(j),err.orderA(j));
end 