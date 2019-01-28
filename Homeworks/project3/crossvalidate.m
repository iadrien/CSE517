function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
% function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
%
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% ktype : (linear, rbf, polynomial)
% Cs   : interval of regularization constant that should be tried out
% paras: interval of kernel parameters that should be tried out
% 
% Output:
% bestC: best performing constant C
% bestP: best performing kernel parameter
% bestval: best performing validation error
% allvalerrs: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)
%
% Trains an SVM classifier for all possible parameter settings in Cs and paras and identifies the best setting on a
% validation split. 
%

%%% Feel free to delete this
% bestC=4.4;
% bestP=4;
% bestval=0.15;

% %% Split off validation data set
% % YOUR CODE
% 
% r = 0.8;
% total_size = size(xTr,2);
% train_size = floor(r*total_size);
% xTr_ = xTr(:,1:train_size);
% yTr_ = yTr(:,1:train_size);
% xTv = xTr(:,train_size+1:end);
% yTv = yTr(:,train_size+1:end);
% %% Evaluate all parameter settings
% % YOUR CODE
% 
% allvalerrs = zeros(size(Cs,2),size(paras,2));
% for i =1:size(Cs,2)
%     for j = 1:size(paras,2)
%      svmclassify =trainsvm(xTr_,yTr_, Cs(1,i),ktype,paras(1,j));
%      allvalerrs(i,j) =sum(sign(svmclassify(xTv))~=yTv(:))/length(yTv);
%     end
% end
% 
% %% Identify best setting
% % YOUR CODE
% 
% [bestval, I] = max(allvalerrs(:));
% [I_row, I_col] = ind2sub(size(allvalerrs),I);
% bestC = Cs(1,I_row);
% bestP = paras(1,I_col);


[~, n] = size(xTr);

if mod(n,5) == 0
    trainRange = n/5;
    testRange = n/5 + 1;
else
    trainRange = floor(n/5);
    testRange = ceiling(n/5);
end

xTrain = xTr(:, 1 : trainRange);
yTrain = yTr(:, 1 : trainRange);
xTest = xTr(:, testRange : n);
yTest = yTr(:, testRange : n);


numC = length(Cs);
numP = length(paras);
allvalerrs = zeros(numC, numP);
min = 1000;
warning('off','all');
optimoptions(@quadprog,'Display','off');
for i = 1 : numC
	for j = 1 : numP
		[svmclassify,~,~]=trainsvm(xTrain,yTrain,Cs(i),'rbf',paras(j));
		allvalerrs(i, j) = sum(sign(svmclassify(xTest))~=yTest(:))/length(yTest);
		if (min > allvalerrs(i, j))
			bestC = Cs(i);
			bestP = paras(j);
			bestval = allvalerrs(i, j);
			min = allvalerrs(i, j);
        end
    end
end
