Data = csvread("trainData.csv",1,4);
trainingInput = Data(:,3:18);
trainingLabelM = Data(:,1);
trainingLabelT = Data(:,2);

CVMdl1 = fitrlinear(trainingInput,trainingLabelM,'Learner','leastsquares','KFold',10);
display(CVMdl1.kfoldLoss);

CVMdl2 = fitrlinear(trainingInput,trainingLabelT,'Learner','leastsquares','KFold',10);
display(CVMdl2.kfoldLoss);
