function pde = datasin1d(approximator)
%% DATACOS1D data for L2-fitting

pde = struct('target',@target,'dtarget',@dtarget,'residual',@residual,...
             'rhs',@rhs,'Dbc',@Dbc);

% exact solution
    function u = target(x)
        u = sin(pi*x);
    end


% 1st order deravative
    function du = dtarget(x)
        du = pi*cos(pi*x);
    end

% residual function : target - approximator
    function r = residual(x)
        r = target(x) - approximator(x);
    end
end