errL2_dnn_2 = load("totalErrL2-DNN-2.mat").totalErrL2;
errL2_snn_1 = load("totalErrL2-SNN-1.mat").totalErrL2;

N = length(errL2_snn_1);
loglog(1:N, errL2_dnn_2,'LineWidth',1.5);
hold on
loglog(1:N, errL2_snn_1,'LineWidth',1.5);
hold on
y = 4*(1:N).^(-1.75);
loglog(1:N,y,'--','LineWidth',1.5);
hold on
legend('L2 error of DNN-2', 'L2 error of SNN-1', 'O(N^{-7/4})');
