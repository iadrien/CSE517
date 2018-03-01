trainningData = csvread("trainData.csv",1,1);

% Using PCA to analyze the raw data
[coeff, score, EV, tsquare, explained] = pca(trainningData(:,6:21),'NumComponents',15,'Economy',false);

display(EV);
display(explained);
plot(EV)
plot(explained)

[row, column] = size(trainningData);

% feature scaling to [0,1]
for i =1:column
    trainningData(:,i) = (trainningData(:,i) - min(trainningData(:,i)))/(max(trainningData(:,i))-min(trainningData(:,i))); 
end

[coeff, score, EV, tsquare, explained] = pca(trainningData(:,6:21),'NumComponents',15,'Economy',false);

display(EV);
display(explained);
plot(EV)
plot(explained)