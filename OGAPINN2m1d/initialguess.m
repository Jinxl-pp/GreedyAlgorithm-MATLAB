function theta = initialguess(quad,quadbc,square,auxInfo,h)
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
B = square(1):(1/1000):square(2);
[ww,bb] = meshgrid(W,B);
w = ww(:); b = bb(:);
A = [w,b];
B = [quad.xqd;ones(size(quad.xqd))];
Bbc = [quadbc.xqd;ones(size(quadbc.xqd))];

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
g = activation(C);
dg = dactivation(Cbc);
d2g = d2activation(C);

%%% preliminary value
f = (auxInfo.f.*quad.wei)';
uk = (auxInfo.uk.*quad.wei)';
duk = (auxInfo.duk.*quadbc.wei)';
d2uk = (auxInfo.d2uk.*quad.wei)';

%%% quadrature value
fg = (-d2g)*f.*w.*w + g*f;
ug = (-d2g)*(-d2uk+uk).*w.*w + g*(-d2uk+uk);
ugbc = dg*duk.*w;

%%% function evaluation on every grid points
loss = -(1/2)*((ug-fg)*(h/2) + ugbc).^2;
idx = find(loss==min(loss));
[i,j] = ind2sub(size(ww),idx(1));
theta = [ww(i,j),bb(i,j)]';
% test;
end