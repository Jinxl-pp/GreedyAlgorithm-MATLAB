%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c
% c    2025, Xianlin Jin
% c
% c    This file tests the convergence rate of orthogonal greedy
% c    algorithm for solving a two-dimensional Allen-Cahn's equation
% c                 ut - lam*uxx + eps*(u^3-u) = 0, 
% c    where x is in (-1,1) and t is in [0,T], with Dirichlet-type 
% c    boundary condition and initial condition as follows:
% c                         u = gD, x = -1 or 1,
% c                         u = u0, t = 0.
% c    The dictionary we use is the P_k^d dictionary for d=2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
addpath('../data/', ...
        '../tool/');

% physical domain 
interval = [0,1];
time = [0,1];
OmegaT = [interval; time];

% quadrature info
quadNum = 100;

% for ReLU-activation
degree = 2;

% exact solution
e = 1e-1;
lam = 1;
pde =  allencahn2d(e,lam);

% training
level = 4;
[errL2, Loss, SNN] = ogaForlayer(level, pde, degree, OmegaT, quadNum);

% save errL2 errL2
% save errE errE

%% display error and convergence order
order = 2.^(1:level)'; % elementwise power 
errL2_order = errL2(order);
showRate(errL2_order, errE_order, order);