function [xpt,ypt,weight] = rmcpts2d(cube,N)
%% RQUADPTS1D quadrature points and weights for every element

%%% mesh decomposition
h = 1/N;
distx = cube(1,2) - cube(1,1);
disty = cube(2,2) - cube(2,1);
xpt = cube(1,1) + distx*rand(1,N);
ypt = cube(2,1) + disty*rand(1,N);
% xqdpts = [xpt; ypt];

%%% weights
weight = repmat(h, 1, N);
end