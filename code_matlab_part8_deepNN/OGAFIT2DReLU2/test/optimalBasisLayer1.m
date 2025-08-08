function theta = optimalBasisLayer1(xqd,wei,square,auxInfo,h)
%% INITIALGUESS finds a good initial guess by enumeration method
% c     Inputs:
% c              xqd: x-axis of quadrature points;
% c           square: range of enumeration for b;
% c          auxInfo: some already calculated value;
% c                h: the mesh size;
% c     Outputs:
% c           theta: (w,b), the parameters of a single neuron
% c    

% c     Take a mesh on the paramter domian {-1,1}*square

W = [-1,1];
% intervals = 500;
intervals = auxInfo.quadInterval;
B = square(1):(1/intervals):square(2);
[ww,bb] = meshgrid(W,B);
w = ww(:); b = bb(:);
A = [w,b];
B = [xqd;ones(size(xqd))];

degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);

% c     Vectorization process for evaluating the objection on every grid points. 
% c     The key matrix  C(i,j) = w(i)*x(j) + b(i)
% c     where
% c     (w(i),b(i)) are samples of parameters, i=1,2,...,N,
% c     (x(j)) are samples of inputs, i=1,2,...,M.

C = A*B;
g = activation(C);

%%% preliminary value
f = (auxInfo.f.*wei)';
uk = (auxInfo.uk.*wei)';

%%% quadrature value
fg = g*f;
ug = g*uk;

%%% function evaluation on every grid points
loss = -(1/2)*((ug-fg)*(h/2)).^2;
idx = find(loss==min(loss));
[i,j] = ind2sub(size(ww),idx(1));
theta = [ww(i,j),bb(i,j)]';
end