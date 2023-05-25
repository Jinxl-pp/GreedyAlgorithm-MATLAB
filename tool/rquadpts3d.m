function [xqdpts,yqdpts,zqdpts,weight] = rquadpts3d(cuboid,numPts,N)
%% RQUADPTS2D quadrature points and weights for every element

%%% mesh decomposition
h = [1/N,1/N,1/N];
hx = h(1);
hy = h(2);
hz = h(1);
x = cuboid(1,1):(1/N):cuboid(1,2); 
y = cuboid(2,1):(1/N):cuboid(2,2); 
z = cuboid(3,1):(1/N):cuboid(3,2); 

%%% from 1d reference element
[pt,wei] = quadpts1d(numPts);
xp = (pt*hx+x(1:end-1)+x(2:end))/2;
yp = (pt*hy+y(1:end-1)+y(2:end))/2;
zp = (pt*hz+z(1:end-1)+z(2:end))/2;

%%% quadrature points
[xpt,ypt,zpt] = meshgrid(xp(:),yp(:),zp(:));
xqdpts = xpt(:)';
yqdpts = ypt(:)';
zqdpts = zpt(:)';

%%% weights
weix = repmat(wei,1,size(xp,2)); weix = weix(:);
weiy = repmat(wei,1,size(yp,2)); weiy = weiy(:);
weiz = repmat(wei,1,size(zp,2)); weiz = weiz(:);
weight = repmat(weix*weiy',1,1,length(weiz));
weight = bsxfun(@times, weight, reshape(weiz,1,1,[]));
weight = weight(:)';
end