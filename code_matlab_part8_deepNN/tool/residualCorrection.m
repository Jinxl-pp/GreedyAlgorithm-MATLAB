function [W2, W1, Bias] = residualCorrection(W2, W1, Bias, degree, paramPreLayer)

%% switch cases
switch degree

    case 1
            W2 = [W2; 1; -1];
            W1 = [W1; paramPreLayer(:,1)'; -paramPreLayer(:,1)'];
            Bias = [Bias; 0; 0];
    case 2
            W2 = [W2; 1/2; 1/2; -1/2; -1/2];
            W1 = [W1; sqrt(2)/2*paramPreLayer(:,1)'; -sqrt(2)/2*paramPreLayer(:,1)';
                      sqrt(2)/2*paramPreLayer(:,1)'; -sqrt(2)/2*paramPreLayer(:,1)'];
            Bias = [Bias; sqrt(2)/2; -sqrt(2)/2;
                         -sqrt(2)/2; sqrt(2)/2];
    case 3
            W2 = [W2; -1/3; 1/3; 1/6; -1/6; 1/6; -1/6];
            W1 = [W1; paramPreLayer(:,1)'; -paramPreLayer(:,1)';
                      paramPreLayer(:,1)'; -paramPreLayer(:,1)';
                      paramPreLayer(:,1)'; -paramPreLayer(:,1)'];
            Bias = [Bias; 0; 0; 1; -1; -1; 1];


end
end