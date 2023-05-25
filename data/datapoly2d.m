function pde = datapoly2d
%% DATAPOLY2D data for the 4th PDE satisfying the zero Neumann BC

pde = struct('rhs',@rhs,'target',@target,...
             'dxxtarget',@dxxtarget,'dxytarget',@dxytarget,...
             'dyytarget',@dyytarget,'laptarget',@laptarget);

% right hand side
    function f = rhs(x, y)
        f = 144*(x.^2 - 1).^2.*(y.^2 - 1).^4 +...
            128*(x.^2 - 1).^3.*(y.^2 - 1).^3 +...
            144*(x.^2 - 1).^4.*(y.^2 - 1).^2 +...
            (x.^2 - 1).^4.*(y.^2 - 1).^4 +...
            384*x.^4.*(y.^2 - 1).^4 +...
            384*y.^4.*(x.^2 - 1).^4 +...
            1152*x.^2.*(x.^2 - 1).*(y.^2 - 1).^4 +...
            1152*y.^2.*(x.^2 - 1).^4.*(y.^2 - 1) +...
            768*x.^2.*(x.^2 - 1).^2.*(y.^2 - 1).^3 +...
            768*y.^2.*(x.^2 - 1).^3.*(y.^2 - 1).^2 +...
            4608*x.^2.*y.^2.*(x.^2 - 1).^2.*(y.^2 - 1).^2;
    end

% exact solution
    function u = target(x, y)
        u = (x.^2 - 1).^4.*(y.^2 - 1).^4;
    end
% dxx of u
    function dxxu = dxxtarget(x, y)
        dxxu = 8*(x.^2 - 1).^3.*(y.^2 - 1).^4 +...
               48*x.^2.*(x.^2 - 1).^2.*(y.^2 - 1).^4;
    end

% dxy of u
    function dxyu = dxytarget(x, y)
        dxyu = 64*x.*y.*(x.^2 - 1).^3.*(y.^2 - 1).^3;
    end

% dyy of u
    function dyyu = dyytarget(x, y)
        dyyu = 8*(x.^2 - 1).^4.*(y.^2 - 1).^3 +...
               48*y.^2.*(x.^2 - 1).^4.*(y.^2 - 1).^2;
    end

% Laplacian u
    function lapu = laptarget(x, y)
        lapu = 8*(x.^2 - 1).^3.*(y.^2 - 1).^4 +...
               8*(x.^2 - 1).^4.*(y.^2 - 1).^3 +...
               48*x.^2.*(x.^2 - 1).^2.*(y.^2 - 1).^4 +...
               48*y.^2.*(x.^2 - 1).^4.*(y.^2 - 1).^2;
    end
end