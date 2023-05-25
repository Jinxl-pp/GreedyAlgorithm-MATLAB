function pde = datacos2d
%% DATACOS2D data for the 2nd PDE satisfying the zero Neumann BC

pde = struct('rhs',@rhs,'target',@target,...
             'dxtarget',@dxtarget,'dytarget',@dytarget);

% right hand side
    function f = rhs(x, y)
        f = cos(2*pi*x).*cos(2*pi*y) + 8*pi^2*cos(2*pi*x).*cos(2*pi*y);
    end

% exact solution
    function u = target(x, y)
        u = cos(2*pi*x).*cos(2*pi*y);
    end

% 1st order deravative dx
    function dxu = dxtarget(x, y)
        dxu = -2*pi*cos(2*pi*y).*sin(2*pi*x);
    end

% 1st order deravative dy
    function dyu = dytarget(x, y)
        dyu = -2*pi*cos(2*pi*x).*sin(2*pi*y);
    end
end