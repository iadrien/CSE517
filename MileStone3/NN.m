trainningData = csvread("trainData.csv",1,0);

[row, column] = size(trainningData);
% feature scaling to [0,1]
for i =7:column
    trainningData(:,i) = (trainningData(:,i) - min(trainningData(:,i)))/(max(trainningData(:,i))-min(trainningData(:,i))); 
end

Y1 = trainningData(:,5);
Y2 = trainningData(:,6);

X = trainningData(:,7:22);

