function pde = discontinuous1d(approximator)
%% DATACOS1D data for L2-fitting

pde = struct('target',@target,'dtarget',@dtarget,'residual',@residual);

% exact solution
    function u = target(x)
        K = 1000;
        u = 1 ./ (1 + exp(-K*(x-0.5)));
    end

% residual function : target - approximator
    function r = residual(x)
        r = target(x) - approximator(x);
    end
end