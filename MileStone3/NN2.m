net = feedforwardnet(16);
net = train(net,X,Y1);
%T1 = net(X);
view(net)