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
rectangle = [0,1;0,1];

% architecture of neural network
layers = [2,20;
          20,100;
          100,1];
nLayers = size(layers, 1);

% the values of parameters
Net.inputDim = 2;
Net.parameters = cell(nLayers,1);
Net.nHiddenLayers = nLayers - 1;
Net.nHiddenNeurons = sum(layers(:,2)) - 1;
Net.nNeuronsEach = layers(1:end-1,2);

% collection of neurons of the previous layer, starting from the input node
Net.preLayer = @(x)(x);

% initializations
degree = 3;             
quadRectangle = 100;
Net.layerIndicator = 0;
Net.approximator = @(x)(0);
pde = datacos2d(Net.approximator);

totalErrL2 = [];
for layer = 1:Net.nHiddenLayers
    fprintf('The training of %d-th layer.\n', layer)
    Net.layerIndicator = layer - 1;
    [errL2, parameters, approximator, Net] = ogaForLayerSkipFirst(Net, degree, pde, rectangle, quadRectangle);
    Net.parameters{layer} = parameters;
    Net.approximator = approximator;
    pde = datacos2d(Net.approximator);
    totalErrL2 = [totalErrL2; errL2];
end

%% both x-axis and y-axis logged
figure(1)
loglog(1:Net.nHiddenNeurons, totalErrL2,'LineWidth',1.5);
hold on
rate = 1/2 + (2*degree+1)/4;
y = 4*(1:Net.nHiddenNeurons).^(-rate);
loglog(1:Net.nHiddenNeurons,y,'--','LineWidth',1.5);
hold on
nNeurons = [1;cumsum(Net.nNeuronsEach)];
loglog(nNeurons(1:end-1), totalErrL2(nNeurons(1:end-1)), 'kd','MarkerSize',7)
legend('L2 error', ['O(N^{-',num2str(rate),'})'], 'first neuron of each layer')


% %% for only the second layer, both x-axis and y-axis logged
% figure(2)
% loglog(1:Net.nNeuronsEach(end), errL2,'LineWidth',1.5);
% hold on
% rate = 1/2 + (2*degree+1)/4;
% y = 5*(1:Net.nNeuronsEach(end)).^(-rate);
% loglog(1:Net.nNeuronsEach(end),y,'--','LineWidth',1.5);
% legend('L2 error', ['O(N^{-',num2str(rate),'})'])

%% end of file