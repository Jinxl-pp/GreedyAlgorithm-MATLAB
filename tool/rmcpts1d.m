function [xqdpts,weight] = rmcpts1d(cube,N)
%% RQUADPTS1D quadrature points and weights for every element

%%% mesh decomposition
h = 1/N;
start = cube(1);
dist = cube(end) - cube(1);
xqdpts = start + dist*rand(1,N);

%%% weights
weight = repmat(h, 1, N);
end