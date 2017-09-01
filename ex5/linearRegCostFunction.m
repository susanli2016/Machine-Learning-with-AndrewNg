function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % nXression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J =(1/(2*m))*(X*theta-y)'*(X*theta-y)+(lambda/(2*m))*sum(theta(2:end).^2);
grad = ((1/m)*(X*theta-y)'*X)';
grad(2:end,:) = grad(2:end,:)+(lambda/m)*theta(2:end,:);









% =========================================================================

grad = grad(:);

end
