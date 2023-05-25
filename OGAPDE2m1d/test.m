%%% preliminary value
f = auxInfo.f;
uk = auxInfo.uk;
duk = auxInfo.duk;
n = length(w);

%%% quadrature value
degree = auxInfo.deg;
activation = @(x)ReLU(x,degree);
dactivation = @(x)dReLU(x,degree,1);
d2activation = @(x)dReLU(x,degree,2);

e = zeros(n,1);
db = zeros(n,1);
for i = 1:n
    theta11 = [w(i),b(i)];

    %%% parameter info
    w11 = theta11(1);
    b11 = theta11(2);

    %%% quadrature value
    % g, dx(g)
    g = activation(w11*xqd+b11);
    dg = dactivation(w11*xqd+b11).*w11;
    % db(g), db(dx(g))
    dbg = dactivation(w11*xqd+b11);
    dbdg = d2activation(w11*xqd+b11).*w11;

    %%% some auxiliary value
    dudg = duk.*dg;
    ug = uk.*g;
    fg = f.*g;

    %%% assemble the integrals
    rg = sum((dudg+ug-fg).*wei)*(h/2);
    E = -(1/2)*(rg)^2;
    e(i) = E;

    %%% assmeble the partial diff of E w.r.t. p,t,and b
    rdbg = sum((duk.*dbdg+uk.*dbg-f.*dbg).*wei)*(h/2);
    dtE = 0;
    dbE = -rg*rdbg;
    db(i) = dbE;
end

%%% loss
figure(1); plot(b(1:4001),loss(1:4001),'r-')
figure(2); plot(b(4002:end),loss(4002:end),'r-')

%%% db(loss)
% figure(3); plot(b(1:2001),db(1:2001),'b-')
% figure(4); plot(b(2002:end),db(2002:end),'b-')
