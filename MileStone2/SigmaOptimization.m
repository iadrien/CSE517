disp ("Training with automatic hyper parameter optimization");
disp(datetime('now'));
gprMdl1T = fitrgp(x,trainingLabelM,'KernelFunction','squaredexponential','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
disp(datetime('now'));
disp("Negative log likelihood is "+gprMdl1M.LogLikelihood);
disp("The training loss is "+resubLoss(gprMdl1T));