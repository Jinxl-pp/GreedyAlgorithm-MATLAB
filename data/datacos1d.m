function pde = datacos1d
%% DATACOS1D data for the 2nd PDE satisfying the zero Neumann BC

pde = struct('rhs',@rhs,'target',@target,'dtarget',@dtarget);

% right hand side
    function f = rhs(x)
        f = pi^2*cos(pi*x) + cos(pi*x);
    end

% exact solution
    function u = target(x)
        u = cos(pi*x);
    end

% 1st order deravative
    function du = dtarget(x)
        du = -pi*sin(pi*x);
    end
end