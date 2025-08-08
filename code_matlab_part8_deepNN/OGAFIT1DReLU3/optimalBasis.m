function theta = optimalBasis(xqd,wei,square,auxInfo,h)
%% INITIALGUESS finds a good initial guess by enumeration method
% c     Inputs:
% c              xqd: x-axis of quadrature points;
% c           square: range of enumeration for b;
% c          auxInfo: some already calculated value;
% c                h: the mesh size;
% c     Outputs:
% c           theta: (w,b), the parameters of a single neuron
% c    

% c     Use randomized dictionary here, for |w|=1 and |b|<=c.
% c     w is generated as normalized Gaussian RVs with zero mean 
% c     value. b is generated with an uniform partition of the
% c     interval [-c,c].

intervals = auxInfo.quadInterval;
b = rand(intervals,1)*diff(square) + square(1);
w = mvnrnd(zeros(auxInfo.dim,1),eye(auxInfo.dim),size(b,1));
% w = rand(size(b,1),auxInfo.dim);
norm2 = repmat(sqrt(sum(w.^2,2)),1,auxInfo.dim);
w = w ./ norm2;

A = [w,b];
B = [auxInfo.p; ones(size(xqd))];

degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);

% c     Vectorization process for evaluating the objection on every grid points. 
% c     The key matrix  C(i,j) = w(i)*x(j) + b(i)
% c     where
% c     (w(i),b(i)) are samples of parameters, i=1,2,...,N,
% c     (x(j)) are samples of inputs, i=1,2,...,M.

C = A*B;
v = activation(C);

%%% preliminary value
uk = (auxInfo.uk.*wei)';
f = (auxInfo.f.*wei)';

%%% quadrature value
u_dot_v = v * uk;
f_dot_v = v * f ;

%%% function evaluation on every grid points
loss = -(1/2)*((u_dot_v - f_dot_v)*(h/2)).^2;
idx = find(loss==min(loss));
theta = A(idx(1),:);
end