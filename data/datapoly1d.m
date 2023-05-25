function pde = datapoly1d
%% DATAPOLY1D data for the 4th PDE satisfying the zero Neumann BC

pde = struct('rhs',@rhs,'target',@target,'d2target',@d2target);

% right hand side
    function f = rhs(x)
        f = x.^8 - 4*x.^6 + 1686*x.^4 - 1444*x.^2 + 145;
    end

% exact solution
    function u = target(x)
        % u = cos(2*pi*x);
        u = (x - 1).^4.*(x + 1).^4;
    end

% 1st order deravative
    function d2u = d2target(x)
        d2u = 8*(x.^2 - 1).^2.*(7*x.^2 - 1);
    end
end