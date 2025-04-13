clc; clear; close all;

%% Data Preparation
% Load and preprocess data
data = readtable('diabetes.csv');
VarNames = {'Preg', 'Gluc', 'BP', 'SkinThick', 'Ins', 'BMI', 'DPF', 'Age'};

% Handle missing values
missingCols = {'Glucose','BloodPressure','SkinThickness','Insulin','BMI'};
for i = 1:length(missingCols)
    col = missingCols{i};
    data.(col)(data.(col) == 0) = NaN;
end
data = rmmissing(data);

% Prepare features/labels
features = table2array(data(:,1:end-1));
labels = categorical(data.Outcome);

% Train-test split (80-20)
rng("default");
cv = cvpartition(size(features,1), 'HoldOut', 0.2);
XTrain = features(cv.training,:);
YTrain = labels(cv.training);
XTest = features(cv.test,:);
YTest = labels(cv.test);

%% Default Decision Tree
% Reference: https://www.mathworks.com/help/stats/decision-trees.html
fprintf('\n=== Default Decision Tree ===\n');
dt_default = fitctree(XTrain, YTrain, 'PredictorNames', VarNames);

% Evaluate
default_train_pred = predict(dt_default, XTrain);
default_test_pred = predict(dt_default, XTest);
default_train_acc = mean(default_train_pred == YTrain);
default_test_acc = mean(default_test_pred == YTest);

fprintf('Training Accuracy: %.2f%%\n', default_train_acc*100);
fprintf('Testing Accuracy: %.2f%%\n', default_test_acc*100);

%% Optimized Decision Tree (Bayesian)
% Reference: https://www.mathworks.com/help/deeplearning/ug/tune-experiment-hyperparameters-using-bayesian-optimization.html
fprintf('\n=== Hyperparameter Optimization ===\n');
params = hyperparameters('fitctree', XTrain, YTrain);
params(1).Range = [1, 100];  % MaxNumSplits
params(2).Range = [1, 50];    % MinLeafSize

opts = struct('Optimizer','bayesopt', 'MaxObjectiveEvaluations',30, ...
              'ShowPlots',true, 'Verbose',1);

dt_optimized = fitctree(XTrain, YTrain, ...
    'OptimizeHyperparameters',params, ...
    'HyperparameterOptimizationOptions',opts, ...
    'PredictorNames',VarNames);

% Evaluate
opt_train_pred = predict(dt_optimized, XTrain);
opt_test_pred = predict(dt_optimized, XTest);
opt_train_acc = mean(opt_train_pred == YTrain);
opt_test_acc = mean(opt_test_pred == YTest);

fprintf('\n=== Optimized Decision Tree Results ===\n');
fprintf('Training Accuracy: %.2f%%\n', opt_train_acc*100);
fprintf('Testing Accuracy: %.2f%%\n', opt_test_acc*100);

%% Comparative Visualization
figure('Position',[100 100 1200 500])

% Default Model
subplot(1,2,1);
confusionchart(YTest, default_test_pred, 'Title','Default Model');
default_imp = predictorImportance(dt_default);

% Optimized Model
subplot(1,2,2);
confusionchart(YTest, opt_test_pred, 'Title','Optimized Model');
opt_imp = predictorImportance(dt_optimized);

%% Feature Importance Comparison
figure;
subplot(1,2,1);
bar(default_imp);
title('Default Model Feature Importance');
xticklabels(VarNames); xtickangle(45);

subplot(1,2,2);
bar(opt_imp);
title('Optimized Model Feature Importance');
xticklabels(VarNames); xtickangle(45);

%% Tree Visualization
figure; view(dt_default,'Mode','graph'); title('Default Tree');
figure; view(dt_optimized,'Mode','graph'); title('Optimized Tree');