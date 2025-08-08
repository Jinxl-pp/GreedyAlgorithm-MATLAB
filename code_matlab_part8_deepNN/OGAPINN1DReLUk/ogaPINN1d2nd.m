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
layers3 = [1,40;
          40,200;
          200,400;
          400,1];

layers2 = [1,40;
          40,600;
          600,1];

layers1 = [1,640;
          640,1];

Layers = {layers1,layers2,layers3};
filenames = {'pinn-relu2-1-640-1-', 'pinn-relu2-1-40-600-1-', 'pinn-relu2-1-40-200-400-1-'};

for i = 1:3

    layers = Layers{i};
    nLayers = size(layers, 1);

    % collection of neurons of the previous layer, starting from the input node
    Net.preLayer = @(x)(x);
    Net.dpreLayer = @(x)(ones(size(x)));
    Net.d2preLayer = @(x)(zeros(size(x)));
    
    % initializations
    degree = 2;            
    quadInterval = 1000;
    Net.layerIndicator = 0;
    Net.approximator.self = @(x)(0);
    Net.approximator.lap = @(x)(0);
    pde = datacos1d(Net.approximator);

    % the values of parameters
    Net.inputDim = 1;
    Net.parameters = cell(nLayers,1);
    Net.nHiddenLayers = nLayers - 1;
    Net.nHiddenNeurons = sum(layers(:,2)) - 1;
    Net.nNeuronsEach = layers(1:end-1,2);


    totalErrL2 = [];
    totalLoss = [];
    for layer = 1:Net.nHiddenLayers
        fprintf('The training of %d-th layer.\n', layer)
        Net.layerIndicator = layer - 1;
        [errL2, Loss, parameters, approximator, Net] = ogaForLayer(Net, degree, pde, interval, quadInterval);
        Net.parameters{layer} = parameters;
        Net.approximator = approximator;
        pde = datacos1d(Net.approximator);
        totalErrL2 = [totalErrL2; errL2];
        totalLoss = [totalLoss; Loss];
    end

    
    fNameErrL2 = ['results_snn_vs_2layer_vs_3layer/', filenames{i}, 'totalErrL2-new1'];
    fNameLoss = ['results_snn_vs_2layer_vs_3layer/', filenames{i}, 'totalLoss-new1'];
    save(fNameErrL2,'totalErrL2');
    save(fNameLoss,'totalLoss');

end

%% plots
addpath('./results_snn_vs_2layer_vs_3layer/')
compare_snn_dnn

% figure(1)
% loglog(1:Net.nHiddenNeurons, totalLoss,'LineWidth',1.5);
% hold on
% rate = (1/2) + (2*(degree-2)+1)/2;
% y = (1:Net.nHiddenNeurons).^(-2*rate);
% loglog(1:Net.nHiddenNeurons,y,'LineWidth',1.5);
% hold on
% nNeurons = [1;cumsum(Net.nNeuronsEach)];
% loglog(nNeurons(1:end-1), totalLoss(nNeurons(1:end-1)), 'kd','MarkerSize',7)
% convergence = ['N^{',num2str(-2*rate),'}'];
% legend('Loss', convergence, 'first neuron of each layer')
% 
% figure(2)
% loglog(1:Net.nHiddenNeurons, totalErrL2,'LineWidth',1.5);
% hold on
% rate = (1/2) + (2*degree+1)/2;
% y = (1:Net.nHiddenNeurons).^(-rate);
% loglog(1:Net.nHiddenNeurons,y,'LineWidth',1.5);
% hold on
% nNeurons = [1;cumsum(Net.nNeuronsEach)];
% loglog(nNeurons(1:end-1), totalErrL2(nNeurons(1:end-1)), 'kd','MarkerSize',7)
% convergence = ['N^{',num2str(-rate),'}'];
% legend('errL2', convergence, 'first neuron of each layer')

%% end of file