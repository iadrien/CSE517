trainningData = csvread("trainData.csv",1,0);

[row, column] = size(trainningData);
% feature scaling to [-1,1]
for i =7:column
    trainningData(:,i) = (trainningData(:,i) - mean(trainningData(:,i)))/(max(trainningData(:,i))-min(trainningData(:,i))); 
end


[coeff, score, EV, tsquare, explained] = pca(trainningData(:,7:22),'NumComponents',15,'Economy',false);

scatter3(score(:,1),score(:,2),score(:,3),10,'r')
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

data1 = transpose(score(1:3,1));
data1 = [data1; transpose(score(1:3,2))];
data1 = [data1; transpose(score(1:3,3))];

figure;
bar(explained)
