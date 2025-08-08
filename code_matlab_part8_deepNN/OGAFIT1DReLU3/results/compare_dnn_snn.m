load('pinn-relu3-1-56-200-1-totalErrL2.mat')
errL2_dnn = totalErrL2;
load('pinn-relu3-1-56-200-1-totalLoss.mat')
loss_dnn = totalLoss;
load('pinn-relu3-1-256-1-totalErrL2.mat')
errL2_snn = totalErrL2;
load('pinn-relu3-1-256-1-totalLoss.mat')
loss_snn = totalLoss;

%% compare the loss curves
N = 256;
degree = 3;

figure(1)
loglog(1:N, loss_dnn,'LineWidth',1.5);
hold on
loglog(1:N, loss_snn,'LineWidth',1.5);
hold on
rate = (1/2) + (2*(degree-2)+1)/2;
y1 = (1:N).^(-2*rate);
loglog(1:N,y1,'LineWidth',1.5);
hold on
y2 = (1:N).^(-4*rate);
loglog(1:N,y2,'LineWidth',1.5);
hold on
nNeurons = 57;
loglog(nNeurons, loss_dnn(nNeurons), 'kd','MarkerSize',7)
convergence1 = ['N^{',num2str(-2*rate),'}'];
convergence2 = ['N^{',num2str(-4*rate),'}'];
legend('Loss-DNN', 'Loss-SNN', convergence, convergence2, 'first neuron of the second hidden layer')


%% compare the L2-error curves
figure(2)
loglog(1:N, errL2_dnn,'LineWidth',1.5);
hold on
loglog(1:N, errL2_snn,'LineWidth',1.5);
hold on
rate = (1/2) + (2*(degree-2)+1)/2;
y = (1:N).^(-2*rate);
loglog(1:N,y,'LineWidth',1.5);
hold on
nNeurons = 57;
loglog(nNeurons, errL2_dnn(nNeurons), 'kd','MarkerSize',7)
convergence = ['N^{',num2str(-2*rate),'}'];
legend('Loss-DNN', 'Loss-SNN', convergence, 'first neuron of the second hidden layer')