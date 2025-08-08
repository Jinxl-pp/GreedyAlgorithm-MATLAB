function pde = datacos1d(approximator)
%% DATACOS1D data for L2-fitting

pde = struct('target',@target,'dtarget',@dtarget,'residual',@residual,...
             'rhs',@rhs,'Dbc',@Dbc);

% exact solution
    function u = target(x)
        u = cos(pi*x);
    end

% right-hand-side function: f = -lap(u) = -u_xx
    function f = rhs(x)
        f = pi^2 * cos(pi*x) + approximator.lap(x);
    end

% Dirichlet's trace g = B(u) = u|_{boundary}
    function g = Dbc(x)
        g = target(x) - approximator.self(x);
    end

% 1st order deravative
    function du = dtarget(x)
        du = -pi*sin(pi*x);
    end

% residual function : target - approximator
    function r = residual(x)
        r = target(x) - approximator.self(x);
    end
end