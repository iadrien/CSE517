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

if nargin<5,tolerance=1e-02;end
% INSERT CODE HERE:

loss = inf;
w = w0;
for t = 0:maxiter
   prevLoss = loss;
   % deepnet minimize
   [loss,gradient]=func(w); 
   
   if loss < prevLoss
       % good still moving down
       stepsize = 1.01*stepsize;
   else 
       stepsize = 0.5*stepsize;
   end
    
   if (norm(gradient)<tolerance) 
       break;
   end
   w = w - stepsize*gradient;
end
end

