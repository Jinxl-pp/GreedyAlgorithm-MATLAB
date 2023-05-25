function [E,dE] = gradientdescent(theta,xqd,yqd,wei,auxInfo,h)
%% GRADIENTDESCENT computes the gradient of the loss function
% c     Inputs:
% c            theta: the parameters of neuron, 
% c                   written in the polar coordinates;
% c              xqd: x-axis of quadrature points;
% c              yqd: y-axis of quadrature points;
% c          auxInfo: some already calculated value;
% c                h: h=(hx,hy) is the mesh size;
% c     Outputs:
% c                E: the value of current loss function;
% c               dE: the gradient of loss function.
% c     

%%% parameter info
t = theta(1);
b = theta(2);
w1 = cos(t);
w2 = sin(t);
hx = h(1); 
hy = h(2);

%%% preliminary value
uk = auxInfo.uk;
dxxuk = auxInfo.dxxuk;
dxyuk = auxInfo.dxyuk;
dyyuk = auxInfo.dyyuk;
f = auxInfo.f;

%%% quadrature value
activation = @(x)ReLU(x,3);
dactivation = @(x)dReLU(x,3,1);
d2activation = @(x)dReLU(x,3,2);
d3activation = @(x)dReLU(x,3,3);
% g, dxx(g) dxy(g) and dyy(g)
g = activation(w1*xqd+w2*yqd+b); 
dxxg = d2activation(w1*xqd+w2*yqd+b).*(w1.*w1);
dxyg = d2activation(w1*xqd+w2*yqd+b).*(w1.*w2);
dyyg = d2activation(w1*xqd+w2*yqd+b).*(w2.*w2);
% dt(g), dt(dxx(g)), dt(dxy(g)) and dt(dyy(g))
dtg = dactivation(w1*xqd+w2*yqd+b).*(-w2*xqd+w1*yqd);
dtdxxg = d3activation(w1*xqd+w2*yqd+b).*(w1.*w1).*(-w2*xqd+w1*yqd)...
       - d2activation(w1*xqd+w2*yqd+b).*(w1.*w2)*2;
dtdxyg = d3activation(w1*xqd+w2*yqd+b).*(w1.*w2).*(-w2*xqd+w1*yqd)...
       + d2activation(w1*xqd+w2*yqd+b).*(w1.^2-w2.^2);
dtdyyg = d3activation(w1*xqd+w2*yqd+b).*(w2.*w2).*(-w2*xqd+w1*yqd)...
       + d2activation(w1*xqd+w2*yqd+b).*(w1.*w2)*2;
% db(g), db(dxx(g)), db(dxy(g)) and db(dyy(g))
dbg = dactivation(w1*xqd+w2*yqd+b);
dbdxxg = d3activation(w1*xqd+w2*yqd+b).*(w1.*w1);
dbdxyg = d3activation(w1*xqd+w2*yqd+b).*(w1.*w2);
dbdyyg = d3activation(w1*xqd+w2*yqd+b).*(w2.*w2);

%%% some auxiliary value
d2ud2g = dxxuk.*dxxg+2*dxyuk.*dxyg+dyyuk.*dyyg; % (1,2,1)
ug = uk.*g;
fg = f.*g;

%%% assemble the integrals
rg = sum((d2ud2g+ug-fg).*wei)*(hx/2)*(hy/2);
E = -(1/2)*(rg)^2;

%%% assmeble the partial diff of E w.r.t. p,t,and b
rdtg = sum((dxxuk.*dtdxxg+2*dxyuk.*dtdxyg+dyyuk.*dtdyyg+uk.*dtg-f.*dtg).*wei)*(hx/2)*(hy/2);
rdbg = sum((dxxuk.*dbdxxg+2*dxyuk.*dbdxyg+dyyuk.*dbdyyg+uk.*dbg-f.*dbg).*wei)*(hx/2)*(hy/2);
dE(1,1) = -rg*rdtg;
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