function [xqdpts,weight] = rquadpts1d(interval,numPts,N)
%% RQUADPTS1D quadrature points and weights for every element

%%% mesh decomposition
h = 1/N;
x = interval(1):(1/N):interval(2); 

%%% from 1d reference element
[pt,wei] = quadpts1d(numPts);
xp = (pt*h+x(1:end-1)+x(2:end))/2;

%%% quadrature points
xqdpts = xp(:)';

%%% weights
weight = repmat(wei,1,size(xp,2));
weight = weight(:)';
end