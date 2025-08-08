%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c
% c    2024 Xianlin Jin
% c
% c    This file tests the convergence of layer-wise orthogonal greedy
% c    algorithm for 2nd-order elliptic problem with PINNs
% c         min_u |-Lap(u)(x)-f(x)|^2_2 + mu*BD(u)^2, x \in (-1,1).
% c    The dictionary we use to build the deep neural network is 
% c    chosen layer-wise, with ReLU as its activation function:
% c                 {ReLU(w*y+b); |w|=1, b \in [-c,c]}.
% c   Here y stands for the neurons from the last layer and is regarded
% c   as the input of this dictionary. Note that the dimension of y 
% c   equals to the number of neurons in the last layer.
% c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
warning off
addpath('../tool','../data')
interval = [0,1];

% architecture of fc-deep neural network
layers = [1,20;
          20,400;
          400,1];

% N = 2^8;
% layers = [1,N;
%           N,1];
nLayers = size(layers, 1);

% the values of parameters
Net.inputDim = 1;
Net.parameters = cell(nLayers,1);
Net.nHiddenLayers = nLayers - 1;
Net.nHiddenNeurons = sum(layers(:,2)) - 1;
Net.nNeuronsEach = layers(1:end-1,2);

% collection of neurons of the previous layer, starting from the input node
Net.preLayer = @(x)(x);
Net.dpreLayer = @(x)(ones(size(x)));
Net.d2preLayer = @(x)(zeros(size(x)));

% initializations
degree = 1;            
quadInterval = 4000;
Net.layerIndicator = 0;
Net.approximator.self = @(x)(0);
Net.approximator.lap = @(x)(0);
pde = datacos1d(Net.approximator);

totalErrL2 = [];
for layer = 1:Net.nHiddenLayers
    fprintf('The training of %d-th layer.\n', layer)
    Net.layerIndicator = layer - 1;
    [errL2, parameters, approximator, Net] = ogaForLayer(Net, degree, pde, interval, quadInterval);
    Net.parameters{layer} = parameters;
    Net.approximator = approximator;
    pde = datacos1d(Net.approximator);
    totalErrL2 = [totalErrL2; errL2];
end

%% plots
figure;
loglog(1:Net.nHiddenNeurons, totalErrL2,'LineWidth',1.5);
hold on
rate = (1/2) + (2*degree+1)/2;
y1 = (1:Net.nHiddenNeurons).^(-rate);
loglog(1:Net.nHiddenNeurons,y1,'LineWidth',1.5);
hold on
y2 = (1:Net.nHiddenNeurons).^(-2*rate);
loglog(1:Net.nHiddenNeurons,y2,'LineWidth',1.5);
hold on
nNeurons = [1;cumsum(Net.nNeuronsEach)];
loglog(nNeurons(1:end-1), totalErrL2(nNeurons(1:end-1)), 'kd','MarkerSize',7)
convergence = ['N^{',num2str(-rate),'}'];
legend('errL2', convergence, 'first neuron of each layer')

%% end of file