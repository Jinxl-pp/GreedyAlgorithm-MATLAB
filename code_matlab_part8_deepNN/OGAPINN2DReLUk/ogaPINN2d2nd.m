%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c
% c    2025 Xianlin Jin
% c
% c    This file tests the convergence of layer-wise orthogonal greedy
% c    algorithm for the Helmholtz equation with PINNs
% c         min_u |Lap(u)(x)+k*u-f(x)|^2_2 + mu*BD(u)^2, x \in (0,1)^2.
% c    The dictionary we use to build the deep neural network is 
% c    chosen layer-wise, with ReLU as its activation function:
% c                 {ReLU(w*y+b); |w|=1, b \in [-c,c]}.
% c    Here y stands for the neurons from the last layer and is regarded
% c    as the input of this dictionary. Note that the dimension of y 
% c    equals to the number of neurons in the last layer.
% c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
warning on
addpath('../tool','../data')
rectangle = [0,1;0,1];

% architecture of fc-deep neural networ
% relu4 with 1-300-1 is working well for lam=1/2
% relu3 with 1-50-200-1 is working well for lam=1 with 4pts quadRectangle=80
layers = [1,100;
          100,100;
          100,1]; 
nLayers = size(layers, 1);

% collection of neurons of the previous layer, starting from the input node
Net.preLayer = @(p)(p);
Net.dxpreLayer = @(p)([1,0;0,0]*ones(size(p)));
Net.dypreLayer = @(p)([0,0;0,1]*ones(size(p)));
Net.lappreLayer = @(p)(zeros(size(p)));

% initializations
degree = 4;            
quadRectangle = 80;
Net.layerIndicator = 0;
Net.approximator.self = @(x)(0);
Net.approximator.helmholtz = @(x)(0);

% PDE information
lam = 1; %1/2
waveNumber = 1; %2*pi/lam;
pde = dataHelmholtz2dsmooth(Net.approximator,waveNumber); % dataHelmholtz2d

% the values of parameters
Net.inputDim = 2;
Net.parameters = cell(nLayers,1);
Net.nHiddenLayers = nLayers - 1;
Net.nHiddenNeurons = sum(layers(:,2)) - 1;
Net.nNeuronsEach = layers(1:end-1,2);

% training process
totalErrL2 = [];
totalLoss = [];
start_time = clock;
for layer = 1:Net.nHiddenLayers
    fprintf('The training of %d-th layer.\n', layer)
    Net.layerIndicator = layer - 1;
    [errL2, Loss, parameters, approximator, Net] = ogaForLayer(Net, degree, pde, rectangle, quadRectangle);
    Net.parameters{layer} = parameters;
    Net.approximator = approximator;
    pde = dataHelmholtz2dsmooth(Net.approximator,waveNumber);
    totalErrL2 = [totalErrL2; errL2];
    totalLoss = [totalLoss; Loss];
end
end_time = clock;
etime(end_time, start_time)

%% plot NN
x = rectangle(1,1):0.005:rectangle(1,2);
y = rectangle(2,1):0.005:rectangle(2,2);
[X,Y] = meshgrid(x,y);
pts = [X(:),Y(:)]';
Z1 = approximator.self(pts);
Z2 = pde.target(pts);
Z1 = reshape(Z1,length(x),length(y));
Z2 = reshape(Z2,length(x),length(y));
figure(1)
mesh(X,Y,Z1)
figure(2)
mesh(X,Y,Z2)

%% plots

figure(3)
loglog(1:Net.nHiddenNeurons, totalLoss,'LineWidth',1.5);
hold on
rate = (1/2) + (2*(degree-2)+1)/4;
y = 5000*(1:Net.nHiddenNeurons).^(-2*rate);
loglog(1:Net.nHiddenNeurons,y,'-.','LineWidth',1.5);
hold on
nNeurons = [1;cumsum(Net.nNeuronsEach)];
loglog(nNeurons(1:end-1), totalLoss(nNeurons(1:end-1)), 'kd','MarkerSize',7)
convergence1 = ['N^{',num2str(-2*rate),'}'];
legend('Loss', convergence1, 'first neuron of each layer')

%% end of file 68