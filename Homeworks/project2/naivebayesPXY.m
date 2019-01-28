function [posprob,negprob] = naivebayesPXY(x,y)
% function [posprob,negprob] = naivebayesPXY(x,y);
%
% Computation of P(X|Y)
% Input:
% x : n input vectors of d dimensions (dxn)
% y : n labels (-1 or +1) (1xn)
%
% Output:
% posprob: probability vector of p(x|y=1) (dx1)
% negprob: probability vector of p(x|y=-1) (dx1)
%

% add one all-ones positive and negative example
[d,n]=size(x);
x=[x ones(d,2)];
y=[y -1 1];

[d,n] = size(x);
%% fill in code here
positiveY = find(y==1);
negativeY = find(y==-1);

positiveXY = sum(x(:,positiveY),2);
negativeXY = sum(x(:,negativeY),2);

posprob = positiveXY/sum(positiveXY);
negprob = negativeXY/sum(negativeXY);
