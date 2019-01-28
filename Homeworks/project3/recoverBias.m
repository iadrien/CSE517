function bias=recoverBias(K,yTr,alphas,C);
% function bias=recoverBias(K,yTr,alphas,C);
%
% INPUT:	
% K : nxn kernel matrix
% yTr : nx1 input labels
% alphas  : nx1 vector or alpha values
% C : regularization constant
% 
% Output:
% bias : the hyperplane bias of the kernel SVM specified by alphas
%
% Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
% 0<alpha<C
%

% mmargins = abs(alphas - C/2);
% 
% index = find(mmargins==max(mmargins));
% 
% bias = yTr(index(1))-(alphas.*yTr)*K(index(1),:);
% YOUR CODE 

mmargins = min(C-alphas,alphas);
index = find(mmargins == max(mmargins));
index = index(1);

bias = yTr(index)-K(index,:)*(alphas.*yTr);