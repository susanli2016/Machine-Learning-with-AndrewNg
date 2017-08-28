function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X*theta);
left = -y'*log(h);
right = (1-y)'*log(1-h);
theta_new = theta(2:size(theta, 1),:);
regularize_1 = 1/(2*m) * lambda * sum(theta_new.^2)
J = (1/m) * sum(left-right) + regularize_1;

regularize_2 = (1/m)*lambda*theta;
grad = (1/m)*X'*(h-y) + regularize_2;
grad(1) = (1/m)*X(:,1)'*(h-y);


% =============================================================

end
