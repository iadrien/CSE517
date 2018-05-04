trainingData = csvread("trainData.csv",1,0);

[row, column] = size(trainingData);
% feature scaling to [-1,1]
for i =7:column
    trainingData(:,i) = (trainingData(:,i) - mean(trainingData(:,i)))/(max(trainingData(:,i))-min(trainingData(:,i))); 
end

x = trainingData(:,7:22);


%% GP1
Kernel = 'squaredexponential';


trainingLabelM = trainingData(:,5);
trainingLossM1 = [];
trainingTimeM1 = [];
for i = 1:10

    t1 = clock;
    gprMdl1M = fitrgp(x,trainingLabelM,'KernelFunction',Kernel,'KernelParameters', [1,1],'Holdout',0.1);
    t2 = clock;

    trainingLossM1 = [trainingLossM1;kfoldLoss(gprMdl1M)];
    trainingTimeM1 = [trainingTimeM1; etime(t2,t1)];
end

trainingLabelT = trainingData(:,6);
trainingLossT1 = [];
trainingTimeT1 = [];
for i = 1:10

    t1 = clock;
    gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction',Kernel,'KernelParameters', [1,1],'Holdout',0.1);
    t2 = clock;

    trainingLossT1 = [trainingLossT1;kfoldLoss(gprMdl1T)];
    trainingTimeT1 = [trainingTimeT1; etime(t2,t1)];
end

%% GP2
Kernel = 'matern32';


trainingLabelM = trainingData(:,5);
trainingLossM2 = [];
trainingTimeM2 = [];
for i = 1:10

    t1 = clock;
    gprMdl1M = fitrgp(x,trainingLabelM,'KernelFunction',Kernel,'KernelParameters', [1,1],'Holdout',0.1);
    t2 = clock;

    trainingLossM2 = [trainingLossM2;kfoldLoss(gprMdl1M)];
    trainingTimeM2 = [trainingTimeM2; etime(t2,t1)];
end

trainingLabelT = trainingData(:,6);
trainingLossT2 = [];
trainingTimeT2 = [];
for i = 1:10

    t1 = clock;
    gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction',Kernel,'KernelParameters', [1,1],'Holdout',0.1);
    t2 = clock;

    trainingLossT2 = [trainingLossT2;kfoldLoss(gprMdl1T)];
    trainingTimeT2 = [trainingTimeT2; etime(t2,t1)];
end



%% Tree


trainingLabelM = trainingData(:,5);
trainingLossMTree = [];
trainingTimeMTree = [];
for i = 1:10

    t1 = clock;
    tree = fitrtree(x,trainingLabelM,'Holdout',0.1);
    t2 = clock;

    trainingLossMTree = [trainingLossMTree;kfoldLoss(tree)];
    trainingTimeMTree = [trainingTimeMTree; etime(t2,t1)];
end

trainingLabelT = trainingData(:,6);
trainingLossTTree = [];
trainingTimeTTree = [];
for i = 1:10

    t1 = clock;
    tree = fitrtree(x,trainingLabelT,'Holdout',0.1);
    t2 = clock;

    trainingLossTTree = [trainingLossTTree;kfoldLoss(tree)];
    trainingTimeTTree = [trainingTimeTTree; etime(t2,t1)];
end


%% SVM
trainingLabelM = trainingData(:,5);
trainingLossMSVM = [];
trainingTimeMSVM = [];
for i = 1:10

    t1 = clock;
    SVM =fitrsvm(x,trainingLabelM,'KernelFunction','gaussian','Holdout',0.1);
    t2 = clock;

    trainingLossMSVM = [trainingLossMSVM;kfoldLoss(SVM)];
    trainingTimeMSVM = [trainingTimeMSVM; etime(t2,t1)];
end

trainingLabelT = trainingData(:,6);
trainingLossTSVM = [];
trainingTimeTSVM = [];
for i = 1:10

    t1 = clock;
    SVM =fitrsvm(x,trainingLabelT,'KernelFunction','gaussian','Holdout',0.1);
    t2 = clock;

    trainingLossTSVM = [trainingLossTSVM;kfoldLoss(SVM)];
    trainingTimeTSVM = [trainingTimeTSVM; etime(t2,t1)];
end

%% LR

trainingLabelM = trainingData(:,5);
trainingLossMLR = [];
trainingTimeMLR = [];
for i = 1:10

    t1 = clock;
    LR =fitrlinear(x,trainingLabelM,'Learner','leastsquares','Holdout',0.1);
    t2 = clock;

    trainingLossMLR = [trainingLossMLR;kfoldLoss(LR)];
    trainingTimeMLR = [trainingTimeMLR; etime(t2,t1)];
end

trainingLabelT = trainingData(:,6);
trainingLossTLR = [];
trainingTimeTLR = [];
for i = 1:10

    t1 = clock;
    LR =fitrlinear(x,trainingLabelT,'Learner','leastsquares','Holdout',0.1);
    t2 = clock;

    trainingLossTLR = [trainingLossTLR;kfoldLoss(LR)];
    trainingTimeTLR = [trainingTimeTLR; etime(t2,t1)];
end

