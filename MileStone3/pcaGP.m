%Data = csvread("trainData.csv",1,4);
x = X(:,1:3);
trainingLabelM = Y1;
trainingLabelT = Y1;

% CVMdl1 = fitrlinear(trainingInput,trainingLabelM,'Learner','leastsquares','KFold',10);
% display(mean(trainingLabelM));
% display(CVMdl1.kfoldLoss);
% 
% CVMdl2 = fitrlinear(trainingInput,trainingLabelT,'Learner','leastsquares','KFold',10);
% display(mean(trainingLabelT));
% display(CVMdl2.kfoldLoss);
% [row, column] = size(x);
% for i =1:column
%     x(:,i) = (x(:,i) - min(x(:,i)))/(max(x(:,i))-min(x(:,i))); 
% end

%% training with motor score as label
%% training without cross validation
disp ("Training with Motor Score");
disp ("Round 1 without cross validation");
disp("Training with squared exponential kernel");
disp(datetime('now'));
gprMdl1T = fitrgp(x,trainingLabelM,'KernelFunction','squaredexponential','KernelParameters', [1,1]);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl1T.LogLikelihood);
disp("The training loss is "+resubLoss(gprMdl1T));
% disp(datetime('now'));
% gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction','squaredexponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Trainning with exponential kernel");
disp(datetime('now'));
gprMdl2T = fitrgp(x,trainingLabelM,'KernelFunction','exponential', 'KernelParameters', [1,1]);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl2T.LogLikelihood);
disp("The traing loss is "+resubLoss(gprMdl2T));
% disp(datetime('now'));
% gprMdl2T = fitrgp(x,trainingLabelT,'KernelFunction','exponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Trainning with matern 3/2 kernel");
disp(datetime('now'));
gprMdl3T = fitrgp(x,trainingLabelM,'KernelFunction','matern32', 'KernelParameters', [1,1]);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl3T.LogLikelihood);
disp("The training loss is "+resubLoss(gprMdl3T));
% disp(datetime('now'));
% gprMdl3T = fitrgp(x,trainingLabelT,'KernelFunction','matern32', 'KernelParameters', [1,1]);
% disp(datetime('now'));


%% training with 10-fold cross validation
disp ("Round 2 with 10-fold cross validation");
disp("Training with squared exponential kernel");
disp(datetime('now'));
gprMdl1T = fitrgp(x,trainingLabelM,'KernelFunction','squaredexponential', 'KernelParameters', [1,1],'Holdout',0.1);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl1T.LogLikelihood);
disp("The loss is "+kfoldLoss(gprMdl1T));
% disp(datetime('now'));
% gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction','squaredexponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Training with exponential kernel");
disp(datetime('now'));
gprMdl2T = fitrgp(x,trainingLabelM,'KernelFunction','exponential', 'KernelParameters', [1,1],'Holdout',0.1);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl2T.LogLikelihood);
disp("The loss is "+kfoldLoss(gprMdl2T));
% disp(datetime('now'));
% gprMdl2T = fitrgp(x,trainingLabelT,'KernelFunction','exponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Training with matern 3/2 kernel");
disp(datetime('now'));
gprMdl3T = fitrgp(x,trainingLabelM,'KernelFunction','matern32', 'KernelParameters', [1,1],'Holdout',0.1);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl3T.LogLikelihood);
disp("The loss is "+kfoldLoss(gprMdl3T));
% disp(datetime('now'));
% gprMdl3T = fitrgp(x,trainingLabelT,'KernelFunction','matern32', 'KernelParameters', [1,1]);
% disp(datetime('now'));


%% training with total score as the label
%% training without cross validation
disp ("Training with Total Score");
disp ("Round 1 without cross validation");
disp("Training with squared exponential kernel");
disp(datetime('now'));
gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction','squaredexponential','KernelParameters', [1,1]);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl1T.LogLikelihood);
disp("The training loss is "+resubLoss(gprMdl1T));
% disp(datetime('now'));
% gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction','squaredexponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Trainning with exponential kernel");
disp(datetime('now'));
gprMdl2T = fitrgp(x,trainingLabelT,'KernelFunction','exponential', 'KernelParameters', [1,1]);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl2T.LogLikelihood);
disp("The traing loss is "+resubLoss(gprMdl2T));
% disp(datetime('now'));
% gprMdl2T = fitrgp(x,trainingLabelT,'KernelFunction','exponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Trainning with matern 3/2 kernel");
disp(datetime('now'));
gprMdl3T = fitrgp(x,trainingLabelT,'KernelFunction','matern32', 'KernelParameters', [1,1]);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl3T.LogLikelihood);
disp("The training loss is "+resubLoss(gprMdl3T));
% disp(datetime('now'));
% gprMdl3T = fitrgp(x,trainingLabelT,'KernelFunction','matern32', 'KernelParameters', [1,1]);
% disp(datetime('now'));


%% training with 10-fold cross validation
disp ("Round 2 with 10-fold cross validation");
disp("Training with squared exponential kernel");
disp(datetime('now'));
gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction','squaredexponential', 'KernelParameters', [1,1],'Holdout',0.1);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl1T.LogLikelihood);
disp("The loss is "+kfoldLoss(gprMdl1T));
% disp(datetime('now'));
% gprMdl1T = fitrgp(x,trainingLabelT,'KernelFunction','squaredexponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Training with exponential kernel");
disp(datetime('now'));
gprMdl2T = fitrgp(x,trainingLabelT,'KernelFunction','exponential', 'KernelParameters', [1,1],'Holdout',0.1);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl2T.LogLikelihood);
disp("The loss is "+kfoldLoss(gprMdl2T));
% disp(datetime('now'));
% gprMdl2T = fitrgp(x,trainingLabelT,'KernelFunction','exponential', 'KernelParameters', [1,1]);
% disp(datetime('now'));

disp("...");
disp("Training with matern 3/2 kernel");
disp(datetime('now'));
gprMdl3T = fitrgp(x,trainingLabelT,'KernelFunction','matern32', 'KernelParameters', [1,1],'Holdout',0.1);
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl3T.LogLikelihood);
disp("The loss is "+kfoldLoss(gprMdl3T));
% disp(datetime('now'));
% gprMdl3T = fitrgp(x,trainingLabelT,'KernelFunction','matern32', 'KernelParameters', [1,1]);
% disp(datetime('now'));
