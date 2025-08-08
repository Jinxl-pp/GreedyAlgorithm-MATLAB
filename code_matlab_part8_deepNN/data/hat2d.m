function pde = hat2d(approximator)
%% DATACOS1D data for L2-fitting

pde = struct('target',@target,'dtarget',@dtarget,'residual',@residual);

% auxiliary functions
g = @(x)(2*ReLU(x,1) - 4*ReLU(x-0.5,1)+ 2*ReLU(x-1,1));
g2 = @(x)(g(g(x)));
r = @(x)(ReLU(x,1)-ReLU(x-1,1));


% exact solution
    function u = target(p)
        x = p(1,:);
        y = p(2,:);
        u = (1/2)*(g2(r(2*(x-0.25))/2) + g2(r(2*(y-0.25))/2) -...
            g2(r(2*(x-0.25))/2+r(2*(y-0.25))/2));
    end

% 1st order deravative
    function du = dtarget(p)
        du = 0;
    end

% residual function : target - approximator
    function r = residual(p)
        r = target(p) - approximator(p);
    end
end