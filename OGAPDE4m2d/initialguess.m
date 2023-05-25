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

T = 0:(pi/20):2*pi; 
B = square(1):(1/20):square(2); 
[tt,bb] = meshgrid(T,B); 
t = tt(:); b = bb(:);
w1 = cos(t);
w2 = sin(t);
A = [w1,w2,b];
B = [gquad.xqd;gquad.yqd;ones(size(gquad.xqd))];

degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);
d2activation = @(x)dReLU(x,degree,2);

% c     Vectorization process for evaluating the objection on every grid points. 
% c     The key matrix  C(i,j) = w1(i)*x(j)+w2(i)*y(j)+b(i)
% c     where
% c     (w1(i),w2(i),b(i)) are samples of parameters, i=1,2,...,N,
% c     (x(j),y(j)) are samples of inputs, i=1,2,...,M.

C = A*B;
g = activation(C);
d2g = d2activation(C);

%%% preliminary value
f = (auxInfo.f.*gquad.wei)';
uk = (auxInfo.uk.*gquad.wei)';
dxxuk = (auxInfo.dxxuk.*gquad.wei)';
dxyuk = (auxInfo.dxyuk.*gquad.wei)';
dyyuk = (auxInfo.dyyuk.*gquad.wei)';

%%% quadrature value
fg = g*f;
ug = g*uk;
d2ud2g1 = d2g*dxxuk.*(w1.*w1); % dxx(uk)*dxx(g)
d2ud2g2 = d2g*dxyuk.*(w1.*w2); % dxy(uk)*dxy(g)
d2ud2g3 = d2g*dyyuk.*(w2.*w2); % dyy(uk)*dyy(g)

%%% function evaluation on every grid points
loss = -(1/2)*((d2ud2g1+2*d2ud2g2+d2ud2g3+ug-fg)*(gquad.area)).^2;
idx = find(loss==min(loss));
[i,j] = ind2sub(size(tt),idx(1));
theta = [tt(i,j),bb(i,j)]';
end