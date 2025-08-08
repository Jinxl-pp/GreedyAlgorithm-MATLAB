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