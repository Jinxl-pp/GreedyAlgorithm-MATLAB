% 
% n=10;
% A = sprand(n,n,0.5);  % Sparse matrix with density 0.5
% A = A'*A;
% b=eye(10,1);
% myA = @(x) afun(x,A);
% [x3] = pcg(@afun,b,1e-8,100);
% 
% 
% function y=afun(x,A)
%     y=A*x;
% end


n = 20;
A = sprand(n,n,0.5);  % Sparse matrix with density 0.5
A = A'*A;
b = ones(20,1);
myA = @(x)afun(x,A);
x1 = pcg(myA,b)

function y = afun(x,A)
y=A*x;
end