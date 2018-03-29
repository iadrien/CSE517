% uses GPML toolkit: http://www.gaussianprocess.org/gpml/code/matlab/doc/

% sample from a GP on a grid of 1000 points in the interval [0, 10]
x_star = linspace(0, 10, 1e3)';

% set mean function, see help meanFunctions for a list

% ?(x) = 0
mean_function = {@meanZero};
theta.mean = []; % no hyperparameters

% ?(x) = c
% mean_function = {@meanConst};
% theta.mean = 1; % hyper parameter is c

% set covariance function, see help covFunctions for a list

% K(x, x') = ?²exp(-½|x - x'|²/?²)
covariance_function = {@covSEiso};
theta.cov = [log(1); log(1)]; % hyperparameters are [log ?; log ?]

% K(x, x') = ?²?(x - x')
% covariance_function = {@covNoise};
% theta.cov = [log(1)]; % hyperparameter is log ?

% convenience function handles
mu = @(varargin) feval(      mean_function{:}, theta.mean, varargin{:});
K  = @(varargin) feval(covariance_function{:}, theta.cov,  varargin{:});

% prior for f(X*)
prior_mean       = mu(x_star);
prior_covariance =  K(x_star, x_star); 

% sometimes need to add a small constant to the diagonal to force positive definiteness
% prior_covariance = prior_covariance + 1e-6 * eye(numel(x_star));
% sometimes need to force the matrix to be symmetric
% prior_covariance = (prior_covariance + prior_covariance') / 2;

% samples = mvnrnd(prior_mean, prior_covariance, 5);

% figure(1);
% clf;
% plot(x_star, samples);
% 
% figure(2);
% clf;
% imagesc(x_star, x_star, prior_covariance); colorbar

% condition on data
Data = csvread("trainData.csv",1,4);
x = Data(:,3:18);
trainingLabelM = Data(:,1);
trainingLabelT = Data(:,2);

% CVMdl1 = fitrlinear(trainingInput,trainingLabelM,'Learner','leastsquares','KFold',10);
% display(mean(trainingLabelM));
% display(CVMdl1.kfoldLoss);
% 
% CVMdl2 = fitrlinear(trainingInput,trainingLabelT,'Learner','leastsquares','KFold',10);
% display(mean(trainingLabelT));
% display(CVMdl2.kfoldLoss);
[row, column] = size(x);
for i =1:column
    x(:,i) = (x(:,i) - min(x(:,i)))/(max(x (:,i))-min(x(:,i))); 
end
y = trainingLabelM;

% observation model, see help likFunctions for a list

% default is p(y | f) = N(y; f, ?²)
theta.lik = log(0.5);
disp(datetime('now'));
% learn hyperparameters by maximizing log p(y | X, ?)
theta = minimize(theta, @gp, -100, [], mean_function, covariance_function, [], ...
                 x, y);

[predictive_mean, predictive_variance] = ...
    gp(theta, [], mean_function, covariance_function, [], x, y, x_star);

predictive_std = sqrt(predictive_variance);
disp(datetime('now'));
% figure(3);
% clf;
% hold('on');
% fill([x_star; flipud(x_star)], ...
%      [predictive_mean + 2 * predictive_std; ...
%       flipud(predictive_mean - 2 * predictive_std)], ...
%      [166, 206, 227] / 255);
% plot(x, y, '.', 'markersize', 55);
% plot(x_star, predictive_mean, ...
%      'color', [31, 120, 180] / 255);
% 
 fprintf('this model gave log marginal likelihood = %0.3f\n', ...
 gp(theta, [], mean_function, covariance_function, [], x, y));