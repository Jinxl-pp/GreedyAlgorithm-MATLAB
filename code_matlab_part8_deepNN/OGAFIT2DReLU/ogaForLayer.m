function [errL2, parameters, uk, Net] = ogaForLayer(Net, degree, pde, rectangle, quadRectangle)
%% initialization and activation functions
uk = @(x)(0);  % u
auxInfo.deg = degree;
activation = @(x)ReLU(x,degree);

%% largest value of L2 norm of neurons from last layer
x = linspace(rectangle(1,1),rectangle(1,2),1e2);
y = linspace(rectangle(2,1),rectangle(2,2),1e2);
[x,y] = meshgrid(x,y);
samples = [x(:)';y(:)'];
c = Net.preLayer(samples);
c = max(sqrt(sum(c.^2,1))) + 1;
c=2;
square = [-c,c];
clear x y

%% quadrature information
% c     Guass quadreture points in 1D: discretize the interval and apply 
% c     2pts Gauss quadrature rule on each element.
% c     Guass quadreture points in 2D: 2D tensor product of 1D 
% c     Guass quadreture points in 3D: 3D tensor product of 1D 
pts = 2;
h = 1/quadRectangle;
[qdpts,wei] = rquadpts2d(rectangle,pts,quadRectangle);
numPts = size(qdpts,2);
auxInfo.quadRectangle = quadRectangle;

%% the previous layer
layer = Net.layerIndicator;
nNeuron = Net.nNeuronsEach(layer+1);

% if the layer indicator is at the input layer
if layer == 0
    inputDim = Net.inputDim;
    trainingSteps = nNeuron;
% if the layer indicator is at one of the hidden layers
else
    paramPreLayer = Net.parameters{layer};
    inputDim = size(paramPreLayer(:,1),1);
    trainingSteps = nNeuron - 2;
end 
auxInfo.dim = inputDim;
auxInfo.preLayer = Net.preLayer;

%% parameters initialization
W2 = zeros(trainingSteps,1);        
W1 = zeros(trainingSteps,inputDim);
Bias = zeros(trainingSteps,1);      
C = zeros(trainingSteps,numPts);

%% Orthogonal greedy algorithm
timeList = zeros(trainingSteps,1);
errL2 = zeros(trainingSteps,1);
for k = 1:trainingSteps

    % c     The subproblem is key to the greedy algorithm, where the inner
    % c     products |(u,g) - (f,g)| should be maximized.
    % c     Part of the inner products can be computed in advance.

    auxInfo.uk = uk(qdpts); 
    auxInfo.f = pde.residual(qdpts);
    if k == 1
        auxInfo.uk = repmat(auxInfo.uk,1,numPts);
    end
    
    % c     The argmax subproblem is difficult to solve even though the
    % c     dimension is not very large. A good initial guess is needed 
    % c     for g and can be given by enumeration. Since |w|=1, only 
    % c     consider w = +-1.

    tic
    start_time = clock;
    theta = optimalBasis(qdpts,wei,square,auxInfo,h);
    fprintf('------------ the N = %d-th neuron ------------ \n', k);
    toc
    end_time=clock;
    timeList(k) = etime(end_time, start_time);
    
    % c     A few steps of training are needed to reach a local minimum 
    % c     near  the initial guess. Gradient descent, coordinate descent,
    % c     Newton's method, etc., can be applied to this problem.

    W1(k,:) = theta(1:end-1);
    Bias(k) = theta(end);
    w = W1(1:k,:);
    b = Bias(1:k);
    
    % c     Computation of the orthogonal projection. The Gram matrix 
    % c     G(i,j) = (g_i, g_j)
    % c     and the RHS vector 
    % c     b(j) = (f, g_j).

    ak = [W1(k,:),Bias(k)];
    B = [Net.preLayer(qdpts);ones(1,size(qdpts,2))];
    ck = ak*B;
    C(k,:) = ck;
    g = activation(C(1:k,:));
    %%%
    Gk = (g*(g.*wei)')*(h/2)*(h/2);                 % gi*gj                 
    %%%
    f = (auxInfo.f.*wei)';
    bk = g*f*(h/2)*(h/2);                           % f*gi
    %%%
    xk = Gk \ bk;
    W2(1:k) = xk;
    cg = xk';
    clear ck bk xk G Gk Gd

    % c     The new projection as the k-th numerical solution. New neuron
    % c     added into the shallow network and coefficients updated for
    % c     all neurons.
    
    uk = @(x)(cg*activation(w*Net.preLayer(x)+b)); 
    
    % c     Computation of numerical errors. Errors in L_2 norm and H^1
    % c     norm will be displayed. The theoretical convergence order in 
    % c     H^m norm is  1/2 + (2(k-m)+1)/2d
    % c     where k is the power of ReLU and d is the dimension.

    l2e = (pde.residual(qdpts) - uk(qdpts)).^2;
    errl2 = sqrt(sum(l2e.*wei)*(h/2)*(h/2));
    errL2(k) = errl2;

    %%% outputs
    fprintf('the L2 fitting error = %e \n', errl2)
    
end

%% add two nodes using the relation x = ReLU(x)-ReLU(-x), for hidden layers
if Net.layerIndicator > 0
    W2 = [W2; 1; -1];
    W1 = [W1; paramPreLayer(:,1)'; -paramPreLayer(:,1)'];
    Bias = [Bias; 0; 0];
end

%% outputs
parameters  = [W2, W1, Bias];
uk = @(x)(W2'*activation(W1*Net.preLayer(x)+Bias)); % x has to be a row vector
l2e = (pde.target(qdpts) - uk(qdpts)).^2;
errl2 = sqrt(sum(l2e.*wei)*(h/2)*(h/2));
if Net.layerIndicator > 0
    errL2 = [errL2; errl2; errl2];
end
fprintf('the L2 fitting error = %e \n', errl2)

%% the vector of neurons of the current layer
Net.preLayer = @(x)(activation(W1*Net.preLayer(x)+Bias)); 

end
