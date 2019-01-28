function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
%
% INPUT:
% func function to minimize
% w0 = initial weight vector 
% stepsize = initial gradient descent stepsize 
% tolerance = if norm(gradient)<tolerance, it quits
%
% OUTPUTS:
% 
% w = final weight vector
%

if nargin<5
    tolerance=1e-02;
end

w = w0;
loss = inf;

for i = 1:maxiter
    [newloss, gradient] = func(w);
    
    if norm(gradient) < tolerance
        break;
    end
    
    if newloss >= loss
        stepsize = stepsize*0.5;
    else
        w = w - stepsize*gradient;
        stepsize = stepsize*1.01;
        loss = newloss;
    end
    
end


