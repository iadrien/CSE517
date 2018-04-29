
trainingInput = X(:,1:3);
trainingLabelM = Y1;
trainingLabelT = Y2;

CVMdl1 = fitrlinear(trainingInput,trainingLabelM,'Learner','leastsquares','KFold',10);
display(mean(trainingLabelM));
display(CVMdl1.kfoldLoss);

CVMdl2 = fitrlinear(trainingInput,trainingLabelT,'Learner','leastsquares','KFold',10);
display(mean(trainingLabelT));
display(CVMdl2.kfoldLoss);
