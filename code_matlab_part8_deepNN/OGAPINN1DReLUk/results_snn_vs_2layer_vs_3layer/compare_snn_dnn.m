% load('pinn-relu2-1-40-200-400-1-totalErrL2-new.mat')
% errL2_dnn3 = totalErrL2;
% load('pinn-relu2-1-40-200-400-1-totalLoss-new.mat')
% loss_dnn3 = totalLoss;
% load('pinn-relu2-1-40-600-1-totalErrL2-new.mat')
% errL2_dnn2 = totalErrL2;
% load('pinn-relu2-1-40-600-1-totalLoss-new.mat')
% loss_dnn2 = totalLoss;
% load('pinn-relu2-1-640-1-totalErrL2-new.mat')
% errL2_snn = totalErrL2;
% load('pinn-relu2-1-640-1-totalLoss-new.mat')
% loss_snn = totalLoss;


load('pinn-relu2-1-40-200-400-1-totalErrL2-great.mat')
errL2_dnn3 = totalErrL2;
load('pinn-relu2-1-40-200-400-1-totalLoss-great.mat')
loss_dnn3 = totalLoss;
load('pinn-relu2-1-40-600-1-totalErrL2-great.mat')
errL2_dnn2 = totalErrL2;
load('pinn-relu2-1-40-600-1-totalLoss-great.mat')
loss_dnn2 = totalLoss;
load('pinn-relu2-1-640-1-totalErrL2-great.mat')
errL2_snn = totalErrL2;
load('pinn-relu2-1-640-1-totalLoss-great.mat')
loss_snn = totalLoss;

%% compare the loss curves
N = 640;
degree = 2;

figure(1)
loglog(1:N, loss_dnn3,'LineWidth',1.5);
hold on
loglog(1:N, loss_dnn2,'LineWidth',1.5);
hold on
loglog(1:N, loss_snn,'LineWidth',1.5);
hold on
rate = (1/2) + (2*(degree-2)+1)/2;
y1 = 30 * (1:N).^(-2*rate);
loglog(1:N,y1,'--','LineWidth',1.0);
hold on
% y2 = (1:N).^(-4*rate);
% loglog(1:N,y2,'LineWidth',1.5);
% hold on
nNeurons = [40, 240];
loglog([nNeurons(1),nNeurons(1)], [loss_dnn3(nNeurons(1)),loss_dnn2(nNeurons(1))], 'k^','MarkerSize',9)
hold on
loglog(nNeurons(2), loss_dnn3(nNeurons(2)), 'ks','MarkerSize',9)

convergence = ['standard convergence rate for ReLU^2: N^{',num2str(-2*rate),'}'];
legend('Loss-DNN, with 3 hidden-layers',...
       'Loss-DNN, with 2 hidden-layers',...
       'Loss-SNN, with 1 hidden-layer',...
       convergence,...
       'first neuron of the 2nd layer', ...
       'first neuron of the 3rd layer');


%% compare the L2-error curves
figure(2)
loglog(1:N, errL2_dnn3,'LineWidth',1.5);
hold on
loglog(1:N, errL2_dnn2,'LineWidth',1.5);
hold on
loglog(1:N, errL2_snn,'LineWidth',1.5);
hold on
rate = (1/2) + (2*(degree)+1)/2;
y1 = (1:N).^(-rate);
loglog(1:N,y1,'--','LineWidth',1.0);
hold on
nNeurons = [40, 200];
loglog([nNeurons(1),nNeurons(1)], [errL2_dnn3(nNeurons(1)),errL2_dnn2(nNeurons(1))], 'k^','MarkerSize',9)
hold on
loglog(nNeurons(2), errL2_dnn3(nNeurons(2)), 'ks','MarkerSize',9)

convergence = ['standard convergence rate for ReLU^2: N^{',num2str(-rate),'}'];
legend('ErrL2-DNN, with 3 hidden-layers',...
       'ErrL2-DNN, with 2 hidden-layers',...
       'ErrL2-SNN, with 1 hidden-layer',...
       convergence,...
       'first neuron of the 2nd layer', ...
       'first neuron of the 3rd layer');