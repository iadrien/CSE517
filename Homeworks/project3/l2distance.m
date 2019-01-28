function D=l2distance(X,Z)
% function D=l2distance(X,Z)
%	
% Computes the Euclidean distance matrix. 
% Syntax:
% D=l2distance(X,Z)
% Input:
% X: dxn data matrix with n vectors (columns) of dimensionality d
% Z: dxm data matrix with m vectors (columns) of dimensionality d
%
% Output:
% Matrix D of size nxm 
% D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
%
% call with only one input:
% l2distance(X)=l2distance(X,X)
%

[d,n]=size(X);


if (nargin==1) % case when there is only one input (X)
	%% fill in code here
    
    D = sqrt(ones(n,1)*diag(X'*X)'+diag(X'*X)*ones(1,n)-X'*X*2);
else  % case when there are two inputs (X,Z)
	%% fill in code here
    [~,m]=size(Z);
    D = sum(X.^2,1)'*ones(1,m)+ones(n,1)*sum(Z.^2,1)-2*X'*Z;
    D(D<0) = 0;
    D = sqrt(D);
end





