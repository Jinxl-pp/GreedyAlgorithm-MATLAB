function theta = initialguess(mcquad,mcquadbc,square,auxInfo)
%% INITIALGUESS finds a good initial guess by enumeration method
% c     Inputs:
% c              xqd: x-axis of quadrature points;
% c              yqd: y-axis of quadrature points;
% c           square: range of enumeration for b;
% c          auxInfo: some already calculated value;
% c                h: h=(hx,hy) is the mesh size;
% c     Outputs:
% c           theta: (t,b), where t is the angle of polar coordinates
% c    


% c     Take a mesh on the paramter domian [0,2pi]*square
% c     reconstruct w1 and w2 with polar coordinates.

T = 0:(pi/50):2*pi; 
B = square(1):(1/50):square(2); 
[tt,bb] = meshgrid(T,B); 
t = tt(:); b = bb(:);
w1 = cos(t);
w2 = sin(t);
A = [w1,w2,b];
B = [mcquad.xqd;mcquad.yqd;ones(size(mcquad.xqd))];
Bbc = [mcquadbc.xqd;mcquadbc.yqd;ones(size(mcquadbc.xqd))];

degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);

% c     Vectorization process for evaluating the objection on every grid points. 
% c     The key matrix  C(i,j) = w1(i)*x(j)+w2(i)*y(j)+b(i)
% c     where
% c     (w1(i),w2(i),b(i)) are samples of parameters, i=1,2,...,N,
% c     (x(j),y(j)) are samples of inputs, i=1,2,...,M.

C = A*B;
Cbc = A*Bbc;
g = activation(C);
dg = dactivation(Cbc);
d2g = d2activation(C);

%%% preliminary value
f = (auxInfo.f.*mcquad.wei)';
uk = (auxInfo.uk.*mcquad.wei)';
dxuk = (auxInfo.dxuk.*mcquadbc.wei.*mcquadbc.indicatorx)';
dyuk = (auxInfo.dyuk.*mcquadbc.wei.*mcquadbc.indicatory)';
dxxuk = (auxInfo.dxxuk.*mcquad.wei)';
dyyuk = (auxInfo.dyyuk.*mcquad.wei)';

%%% quadrature value
dxg = dg.*mcquadbc.indicatorx;
dyg = dg.*mcquadbc.indicatory;
fg = (-d2g)*f.*(w1.^2) + ...
     (-d2g)*f.*(w2.^2) + g*f;
ug = (-d2g)*(-dxxuk-dyyuk+uk).*(w1.^2) + ...
     (-d2g)*(-dxxuk-dyyuk+uk).*(w2.^2) + g*(-dxxuk-dyyuk+uk);
ugbc = dxg*dxuk.*w1 + dxg*dyuk.*w1 + ...
       dyg*dxuk.*w2 + dyg*dyuk.*w2;

%%% function evaluation on every grid points
loss = -(1/2)*(ug-fg+ugbc).^2;
idx = find(loss==min(loss));
[i,j] = ind2sub(size(tt),idx(1));
theta = [tt(i,j),bb(i,j)]';
end