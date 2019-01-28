function logratio = naivebayes(x,y,x1)
% function logratio = naivebayes(x,y);
%
% Computation of log P(Y|X=x1) using Bayes Rule
% Input:
% x : n input vectors of d dimensions (dxn)
% y : n labels (-1 or +1)
% x1: input vector of d dimensions (dx1)
%
% Output:
% logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))

[d,n] = size(x);
%% fill in code here
[pos,neg] = naivebayesPY(x,y);
[posprob, negprob] = naivebayesPXY(x,y);

poscond = log(pos) + dot(x1, log(posprob));
negcond = log(neg) + dot(x1, log(negprob));

logratio = poscond - negcond;