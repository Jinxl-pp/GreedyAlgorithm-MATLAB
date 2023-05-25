function [xqdpts,yqdpts,weight] = rquadpts2d(rectangle,numPts,N)
%% RQUADPTS2D quadrature points and weights for every element

%%% mesh decomposition
h = [1/N,1/N];
hx = h(1);
hy = h(2);
x = rectangle(1,1):(1/N):rectangle(1,2); 
y = rectangle(2,1):(1/N):rectangle(2,2); 

%%% from 1d reference element
[pt,wei] = quadpts1d(numPts);
xp = (pt*hx+x(1:end-1)+x(2:end))/2;
yp = (pt*hy+y(1:end-1)+y(2:end))/2;

%%% quadrature points
[xpt,ypt] = meshgrid(xp(:),yp(:));
xqdpts = xpt(:)';
yqdpts = ypt(:)';

%%% weights
weix = repmat(wei,1,size(xp,2));
weiy = repmat(wei,1,size(yp,2));
weix = weix(:);
weiy = weiy(:);
weight = weix * weiy';
weight = weight(:)';
end