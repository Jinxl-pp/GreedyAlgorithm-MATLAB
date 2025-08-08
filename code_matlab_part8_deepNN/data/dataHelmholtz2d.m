function pde = dataHelmholtz2d(approximator,k)
%% DATACOS1D data for L2-fitting
% wn for wave number

pde = struct('target',@target,'source',@source,'Dbc',@Dbc,'residual',@residual,'wn',k);

% exact solution
    function u = target(p)
        x = p(1,:);
        y = p(2,:);
        u = sin(k*x).*sin(k*y);
    end

% source term
    function rhs = source(p)
        x = p(1,:);
        y = p(2,:);
        rhs = -k^2.*sin(k*x).*sin(k*y) - approximator.helmholtz(p);
    end

% Dirichlet's trace g = B(u) = u|_{boundary}
    function g = Dbc(p)
        g = target(p) - approximator.self(p);
    end

% residual function : target - approximator
    function r = residual(p)
        r = target(p) - approximator.self(p);
    end
end