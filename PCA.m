trainningData = csvread("trainData.csv",1,1);

% Using PCA to analyze the raw data
[coeff, score, EV, tsquare, explained] = pca(trainningData(:,7:21),'NumComponents',15,'Economy',false);

display(EV);
display(explained);