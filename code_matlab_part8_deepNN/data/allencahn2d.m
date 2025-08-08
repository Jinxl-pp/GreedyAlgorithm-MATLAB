function pde = allencahn2d(e, lam)
%% DATACOS1D data for L2-fitting

pde = struct('target',@target,'rhs',@rhs,'residual',@residual);

% exact solution
    function u = target(p)
        c1 = sqrt((e/8/lam));
        c2 = 3*e/4;
        x = p(1,:);
        t = p(2,:);
        u = -(1/2) + (1/2)*tanh(c1*x - c2*t);
    end

% source term function
    function f = rhs(p)
        x = p(1,:);
        f = zeros(size(x));
    end


% residual function : target - approximator
    function r = residual(p)
        r = target(p);
    end
end