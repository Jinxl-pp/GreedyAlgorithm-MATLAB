function theta = initialguess(gquad,square,auxInfo)
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
B = [gquad.xqd;gquad.yqd;ones(size(gquad.xqd))];

degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);

% c     Vectorization process for evaluating the objection on every grid points. 
% c     The key matrix  C(i,j) = w1(i)*x(j)+w2(i)*y(j)+b(i)
% c     where
% c     (w1(i),w2(i),b(i)) are samples of parameters, i=1,2,...,N,
% c     (x(j),y(j)) are samples of inputs, i=1,2,...,M.

C = A*B;
g = activation(C);
dg = dactivation(C);

%%% preliminary value
f = (auxInfo.f.*gquad.wei)';
uk = (auxInfo.uk.*gquad.wei)';
dxuk = (auxInfo.dxuk.*gquad.wei)';
dyuk = (auxInfo.dyuk.*gquad.wei)';

%%% quadrature value
fg = g*f;
ug = g*uk;
dudg1 = dg*dxuk.*w1; % dx(uk)*dx(g)
dudg2 = dg*dyuk.*w2; % dy(uk)*dy(g)

%%% function evaluation on every grid points
loss = -(1/2)*((dudg1+dudg2+ug-fg)*(gquad.area)).^2;
idx = find(loss==min(loss));
[i,j] = ind2sub(size(tt),idx(1));
theta = [tt(i,j),bb(i,j)]';
end