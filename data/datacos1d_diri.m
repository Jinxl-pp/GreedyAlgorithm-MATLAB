function pde = datacos1d_diri
%% DATACOS1D data for the 2nd-order PDE satisfying the zero Dirichlet BC

pde = struct('rhs',@rhs,'target',@target,'dtarget',@dtarget);

% right hand side
    function f = rhs(x)
        f = pi^2/4*cos(pi/2*x) + cos(pi/2*x);
    end

% exact solution
    function u = target(x)
        u = cos(pi/2*x);
    end

% 1st order deravative
    function du = dtarget(x)
        du = -pi/2*sin(pi/2*x);
    end
end