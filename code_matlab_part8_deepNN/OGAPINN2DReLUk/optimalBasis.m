function theta = optimalBasis(qdpts,wei,xqdbc,weibc,square,auxInfo,h)
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

rectangles = auxInfo.quadRectangle * 1;
% rectangles = 100;
b = rand(rectangles,1)*diff(square) + square(1);
w = mvnrnd(zeros(auxInfo.dim,1),eye(auxInfo.dim),size(b,1));
% w = rand(size(b,1),auxInfo.dim);
norm2 = repmat(sqrt(sum(w.^2,2)),1,auxInfo.dim);
w = w ./ norm2;

A = [w,b];
B = [auxInfo.p; ones(1,size(qdpts,2))];
Bbc = [auxInfo.pbc; ones(1,size(xqdbc,2))];

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
v = activation(C);
dv = dactivation(C);
vbc = activation(Cbc);
d2v = d2activation(C);

%%% preliminary value
mu = auxInfo.mu;
wn = auxInfo.wn;
f = (auxInfo.f.*wei)';
huk = (auxInfo.huk.*wei)';
ukbc = (auxInfo.ukbc.*weibc)';
gDbc = (auxInfo.gD.*weibc)';

%%% quadrature value
hu_dot_hv = (dv.*(w*auxInfo.lapp))*huk + (d2v.*((w*auxInfo.dxp).^2+(w*auxInfo.dyp).^2))*huk + wn * v*huk;
f_dot_hv = (dv.*(w*auxInfo.lapp))*f + (d2v.*((w*auxInfo.dxp).^2+(w*auxInfo.dyp).^2))*f + wn * v*f;
umg_dot_v_bc = mu * vbc * (ukbc - gDbc);

%%% function evaluation on every grid points
loss = -(1/2)*((hu_dot_hv - f_dot_hv)*(h/2)*(h/2) + umg_dot_v_bc*(h/2)).^2;
idx = find(loss==min(loss));
theta = A(idx(1),:);
end