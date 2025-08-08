function [qdbc, weibc] = rquadpts2dbc(rectangle, pts, quadRectangle)

% rectangle = [x1,x2; y1,y2]:
%   a(x1,y1) --- b(x1,y2)
%       |            |
%   c(x2,y1) --- d(x2,y2)

    x1 = rectangle(1,1); x2 = rectangle(1,2);
    y1 = rectangle(2,1); y2 = rectangle(2,2);
        
    interval = [0,1];
    [qd_ref,wei_ref] = rquadpts1d(interval,pts,quadRectangle);

    xqd = qd_ref*(x2-x1) + x1;
    yqd1 = y1 * ones(size(qd_ref));
    yqd2 = y2 * ones(size(qd_ref));

    yqd = qd_ref*(y2-y1) + y1;
    xqd1 = x1 * ones(size(qd_ref));
    xqd2 = x2 * ones(size(qd_ref));

    xqdbc = [xqd,xqd,xqd1,xqd2];
    yqdbc = [yqd1,yqd2,yqd,yqd];
    
    qdbc = [xqdbc; yqdbc];
    weibc = repmat(wei_ref,1,4);

end