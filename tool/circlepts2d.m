function [xmcpts,ymcpts,weight] = circlepts2d(cen, R, N)
r = R * sqrt(rand(1,N));
t = rand(1,N)*2*pi;
xmcpts = r.*cos(t) + cen(1);
ymcpts = r.*sin(t) + cen(2);
weight = pi * R^2;
weight = weight / N;
end