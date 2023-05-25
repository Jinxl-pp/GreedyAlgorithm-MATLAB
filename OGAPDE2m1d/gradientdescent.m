function [E,dE] = gradientdescent(theta,xqd,wei,auxInfo,h)
%% GRADIENTDESCENT computes the gradient of the loss function
% Inputs:
%        theta: the parameters of neuron, 
%               written in the polar coordinates;
%          xqd: x-axis of quadrature points;
%      auxInfo: some already calculated value;
%            h: the mesh size;
% Outputs:
%            E: the value of current loss function;
%           dE: the gradient of loss function.
%

%%% parameter info
w = theta(1);
b = theta(2);

%%% preliminary value
f = auxInfo.f;
uk = auxInfo.uk;
duk = auxInfo.duk;

%%% quadrature value
degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);
% g, dx(g)
g = activation(w*xqd+b); 
dg = dactivation(w*xqd+b).*w;
% db(g), db(dx(g))
dbg = dactivation(w*xqd+b);
dbdg = d2activation(w*xqd+b).*w;

%%% some auxiliary value
dudg = duk.*dg;
ug = uk.*g;
fg = f.*g;

%%% assemble the integrals
rg = sum((dudg+ug-fg).*wei)*(h/2);
E = -(1/2)*(rg)^2;

%%% assmeble the partial diff of E w.r.t. p,t,and b
rdbg = sum((duk.*dbdg+uk.*dbg-f.*dbg).*wei)*(h/2);
dE(1,1) = 0;
dE(2,1) = -rg*rdbg;
end

%% ReLU
function y=ReLU(x,n)
y=max(0,x).^n;
end

%% derivatives of ReLU
function y=dReLU(x,n,m)
y1=zeros(size(x));
y1(x>0)=1;
y=factorial(n)/factorial(n-m)*max(0,x).^(n-m).*y1.^m;
end

%% sigmoid 
function y = sigmoid(x)
y = 1./(1+exp(-x));
end

%% derivatives of sigmoid
function y = dsigmoid(x,n)
switch n
    case 1
        y = exp(-x)./(exp(-x) + 1).^2;
        
    case 2
        y = (2*exp(-2*x))./(exp(-x) + 1).^3 ...
            - exp(-x)./(exp(-x) + 1).^2;
    case 3
        y = exp(-x)./(exp(-x) + 1).^2 ...
            - (6*exp(-2*x))./(exp(-x) + 1).^3 ...
            + (6*exp(-3*x))./(exp(-x) + 1).^4;
end
end