syms x
w = [-1,1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,1];
w = repmat(w,1,10);x
b = 2*(rand(1,length(w))-0.5);
f = 0;
for i = 1:length(w)
    y = w(i)*x + b(i);
    f = f + (y/2 + abs(y)/2)^2;
end
Iexact = double(int(f,-1,1));

numPts = 2;
[pt,wei] = quadpts1d(numPts);
x = [-1,sort(-b./w),1];
h = diff(x);
xp = (pt*h+x(1:end-1)+x(2:end))/2;
xqdpts = xp(:)';
h = repmat(h,numPts,1);
h = h(:)';
weight = repmat(wei,1,size(xp,2));
weight = weight(:)';

F = matlabFunction(f);
Inum = F(xqdpts).*weight.*h;
Inum = double(sum(Inum)/2);

err = Inum - Iexact
