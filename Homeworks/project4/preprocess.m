function [xTr,xTe,u,m]=preprocess(xTr,xTe)
% function [xTr,xTe,u,m]=preprocess(xTr,xTe);
%
% Preproces the data to make the training features have zero-mean and
% standard-deviation 1
%
% output:
% xTr - pre-processed training data 
% xTe - pre-processed testing data
% 
% u,m - any other data should be pre-processed by x-> u*(x-m)
%

% % 

m=mean(xTr,2);
u=diag(1./std(xTr,0,2)');

xTr=u*(xTr-m);
xTe=u*(xTe-m);

end





