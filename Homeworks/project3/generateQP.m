function [H,q,Aeq,beq,lb,ub]=generateQP(K,yTr,C);
% function [H,q,A,b,lb,ub]=generateQP(K,yTr,C);
%
% INPUT:	
% K : nxn kernel matrix
% yTr : 1xn input labels
% C : regularization constant
% 
% Output:
% H,q,Aeq,beq,lb,ub as defined in the MATLAB function 'quadprog'
%
% A call of quadprog (H, q, A, b,Aeq,beq,lb, ub,alpha0); should return the solution of the SVM specified by K,yTr,C
% for any valid alpha0 initialization.
%

[d,n]=size(K);
assert(d==n);

% YOUR CODE

% yTr K yTr;
% -ones(n,1);
% yTr;
% zeros(n,1);
% zeros(n,1);
% ones(n,1)*C;

H = K.* (yTr' * yTr);
q = -ones(n,1);
Aeq = yTr;
beq = 0;
lb = zeros(n,1);
ub = C * ones(n,1);

