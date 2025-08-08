function theta = optimalBasis(xqd,wei,xqdbc,weibc,square,auxInfo,h)
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
Bbc = [auxInfo.pbc; ones(size(xqdbc))];

degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);

% c     Vectorization process for evaluating the objection on every grid points. 
% c     The key matrix  C(i,j) = w(i)*x(j) + b(i)
% c     where
% c     (w(i),b(i)) are samples of parameters, i=1,2,...,N,
% c     (x(j)) are samples of inputs, i=1,2,...,M.

C = A*B;
Cbc = A*Bbc;
dv = dactivation(C);
vbc = activation(Cbc);
d2v = d2activation(C);

%%% preliminary value
mu = auxInfo.mu;
f = (auxInfo.f.*wei)';
d2uk = (auxInfo.d2uk.*wei)';
ukbc = (auxInfo.ukbc.*weibc)';
gDbc = (auxInfo.gD.*weibc)';

%%% quadrature value
g_dot_v_bc = mu * vbc * gDbc;
u_dot_v_bc = mu * vbc * ukbc;
f_dot_lapv = (d2v.*(w*auxInfo.dp).*(w*auxInfo.dp))*f  + (dv.*(w*auxInfo.d2p))*f ;
lapu_dot_lapv = (d2v.*(w*auxInfo.dp).*(w*auxInfo.dp))*d2uk + (dv.*(w*auxInfo.d2p))*d2uk;

%%% function evaluation on every grid points
loss = -(1/2)*((lapu_dot_lapv + f_dot_lapv)*(h/2) + u_dot_v_bc - g_dot_v_bc).^2;
idx = find(loss==min(loss));
theta = A(idx(1),:);
end