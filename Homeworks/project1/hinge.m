function [loss,gradient,preds]=hinge(w,xTr,yTr,lambda)
% function w=ridge(xTr,yTr,lambda)
%
% INPUT:
% xTr dxn matrix (each column is an input vector)
% yTr 1xn matrix (each entry is a label)
% lambda regression constant
% w weight vector (default w=0)
%
% OUTPUTS:
%
% loss = the total loss obtained with w on xTr and yTr
% gradient = the gradient at w
%

[d,n]=size(xTr);

regulerizer = lambda * dot(w,w);
wTx = w' * xTr;
loss = sum( max( 1 - yTr.* wTx, 0 ) ) + regulerizer;

yTr( yTr .* wTx > 1 ) = 0;
gradient =  - xTr * yTr' + 2 * lambda * w;

preds = w'*xTr;