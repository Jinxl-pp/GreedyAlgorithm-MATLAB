%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c
% c    2024 Xianlin Jin
% c
% c    This file tests the convergence of layer-wise orthogonal greedy
% c    algorithm for L2 data-fitting problem
% c                     min_u |u(x)-f(x)|_2, x \in (-1,1).
% c    The dictionary we use to build the deep neural network is 
% c    chosen layer-wise, with ReLU as its activation function:
% c                 {ReLU(w*y+b); |w|=1, b \in [-c,c]}.
% c   Here y stands for the neurons from the last layer and is regarded
% c   as the input of this dictionary. Note that the dimension of y 
% c   equals to the number of neurons in the last layer.
% c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
addpath('../tool','../data')
interval = [-1,1];

% initializations
degree = 1;              % degree of ReLU
nNeuron = 10;           % number of neurons of layer1
quadInterval = 500;
approximator = @(x)(0);
pde = datacos1d(approximator);
[errL2_layer1, paramLayer1, u] = OGALayer1(nNeuron, degree, pde, interval, quadInterval);

% the second-layer dictionary approximation
degree = 1;              % degree of ReLU
nNeuron = 500;           % number of neurons of layer2
quadInterval = 1000;
pde = datacos1d(u);
[errL2_layer2, paramLayer2, u] = OGALayer2(paramLayer1, nNeuron, degree, pde, interval, quadInterval);

errL2 = [errL2_layer1; errL2_layer2];

N = 510;
loglog(1:N,errL2)
hold on
y = (1:N).^(-2);
loglog(1:N,y)
%% test plot
% figure(1)
% x = -1:0.001:1;
% plot(x,u(x),'r-')
% hold on
% plot(x,pde.target(x),'b-')
% legend('approximation','target')
% 
% figure(2)
% x = -1:0.001:1;
% u_equal = ReLU(u(x),1) - ReLU(-u(x),1);
% plot(x,u_equal,'r-')
% hold on
% plot(x,pde.target(x),'b-')
% legend('approximation','target')
%% end of file