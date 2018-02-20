function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];

z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

cost = 0;
for i=1:m
    y_m = zeros(num_labels, 1);
    y_m(y(i)) = 1;
    h = (a3(i, :))';

    cost1 = y_m' * log(h); 
    cost2 = (1 - y_m)' * log(1 - h);
    cost = cost + sum(cost1 + cost2);

    %back propagation
    err3 = h - y_m;
    z2_m = z2(i, :)';
    temp = Theta2'*err3;
    temp = temp(2:end);
    err2 = temp.*sigmoidGradient(z2_m);
    a2_m = a2(i,:)';
    a1_m = X(i,:)';
    delta2 = err3 * a2_m';
    delta1 = err2 * a1_m';

    Theta1_grad = Theta1_grad + delta1./m;
    Theta2_grad = Theta2_grad + delta2./m;
end

Theta1_reg = Theta1;
Theta2_reg = Theta2;

penalty1 = 0;
penalty2 = 0;

Theta1_reg(:,1) = zeros(size(Theta1_reg, 1),1); 
Theta2_reg(:,1) = zeros(size(Theta2_reg, 1),1); 

p1 = size(Theta1_reg, 1);
p2 = size(Theta2_reg, 1);

for w=1:p1
    penalty1 = penalty1 + Theta1_reg(w,:)*Theta1_reg(w,:)';
end

for w=1:p2
    penalty2 = penalty2 + Theta2_reg(w,:)*Theta2_reg(w,:)';
end

penalty = (lambda/(2*m)) * (penalty1 + penalty2);

J = (-1/m) * cost + penalty;

grad1_p = (lambda/m)*Theta1_reg;
grad2_p = (lambda/m)*Theta2_reg;

Theta1_grad = Theta1_grad + grad1_p;
Theta2_grad = Theta2_grad + grad2_p;

grad = [Theta1_grad(:); Theta2_grad(:)];

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
