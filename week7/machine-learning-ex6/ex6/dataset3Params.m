function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_trial = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_trial = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

%C_trial = [0.001, 0.1, 30];
%sigma_trial = [0.001, 0.1, 30];

%C_trial = [1];
%sigma_trial = [1];

size_trial = size(C_trial, 2);
size_pred = size_trial^2;

pred_matrix = zeros(size_pred, 3);

k = 1; % index for prediction matrix

%fprintf('PSG DBG: starting the loop \n');

%disp(size(X));
%disp(size(y));

for i=1:size_trial
    for j=1:size_trial
        model = svmTrain(X, y, C_trial(i), @(X, y) gaussianKernel(X, y, sigma_trial(j)));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        pred_matrix(k, 1) = err;
        pred_matrix(k, 2) = C_trial(i);
        pred_matrix(k, 3) = sigma_trial(j);
        k = k + 1;
    end
end

%fprintf('PSG DBG: ending the loop \n');

[value, index] = min(pred_matrix(:,1));

C = pred_matrix(index, 2);
sigma = pred_matrix(index, 3);





% =========================================================================

end
