function Theta = FibonacciLattices(N)
%% This function generates the Fibonacci Lattices on the 3D unit sphere S^3.
% c     Theta: the returned value, sized N-by-3.
% c            Each row of theta is a sample on S^3.

%%% lattices on the unit 2D rectangle [0,1]^2
phi = (1+sqrt(5))/2;
index = (0:2*N-1)';
XY = [mod(index/phi,1), index/(2*N-1)];

%%% cylindrical equal-area projection
TP = [2*pi*XY(:,1), acos(1-2*XY(:,2))];
TP = TP(1:N,:);

%%% 3D polar coordinates
Theta = [cos(TP(:,1)).*sin(TP(:,2)), ...
         sin(TP(:,1)).*sin(TP(:,2)), ...
         cos(TP(:,2))];

end