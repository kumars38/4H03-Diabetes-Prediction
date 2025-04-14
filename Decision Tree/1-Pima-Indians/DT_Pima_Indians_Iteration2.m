clear variables;
clc;
close all;

%% Import Predictors/Targets
data = readmatrix("pima-indians-diabetes.csv", 'NumHeaderLines', 1); % read from csv
predictors = data(:,1:end-1);
target = data(:,end);
VarName = {'Preg'; 'Glucose'; 'Blood Pressure'; 
    'Skin Thickness'; 'Insulin';
    'BMI'; 'DPF'; 'Age'};

%% Pre-Processing

% Identify columns with missing values
columns_to_check = [2, 3, 4, 5, 6]; % Glucose, BP, Skin, Insulin, BMI
column_names = {'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI'};

% Identify rows with missing values
has_missing = false(size(predictors, 1), 1);
for i = 1:length(columns_to_check)
    col = columns_to_check(i);
    zero_indices = predictors(:, col) == 0;
    has_missing = has_missing | zero_indices;
end

% Remove rows with missing values
complete_cases = ~has_missing;

% Create clean dataset
features = predictors(complete_cases, :); 
labels = target(complete_cases);          

% Calculate the number of samples
NumSamples = size(features, 1);         

% Apply normalization to features (similar to ANN)
features_norm = normalize(features, 'range');

% Random seed for reproducibility - use same seed as ANN
rng(999);

%% Cross-validation Setup (similar to ANN)
K = 5;
cv = cvpartition(NumSamples, 'KFold', K);

% Init arrays for metrics - both test and train
test_accuracies = zeros(K, 1);
test_precisions = zeros(K, 1);
test_recalls = zeros(K, 1);
train_accuracies = zeros(K, 1);
train_precisions = zeros(K, 1);
train_recalls = zeros(K, 1);

% Combined confusion matrices over all folds
testCombinedConfMat = zeros(2, 2);
trainCombinedConfMat = zeros(2, 2);

for k = 1:K
    fprintf('Fold %d\n', k);
    
    % Get training and test indices
    trainIdx = training(cv, k);
    testIdx = test(cv, k);
    
    % Extract training and testing data
    xTrain = features_norm(trainIdx, :);
    yTrain = labels(trainIdx);
    xTest = features_norm(testIdx, :);
    yTest = labels(testIdx);
    
    % Create decision tree with optimized hyperparameters
    dt = fitctree(xTrain, yTrain, ...
        'MinLeafSize', 8, ...        % Prevent overfitting with larger leaf size
        'MaxNumSplits', 15, ...      % Control tree complexity
        'SplitCriterion', 'gdi');    % Gini's diversity index
    
    % Predict on test data
    yTestPred = predict(dt, xTest);
    
    % Predict on training data
    yTrainPred = predict(dt, xTrain);
    
    % Compute TEST confusion matrix
    testConfMat = confusionmat(yTest, yTestPred);
    testCombinedConfMat = testCombinedConfMat + testConfMat;
    
    % Compute TRAIN confusion matrix
    trainConfMat = confusionmat(yTrain, yTrainPred);
    trainCombinedConfMat = trainCombinedConfMat + trainConfMat;
    
    % TEST Metrics
    TN_test = testConfMat(1,1);
    FN_test = testConfMat(2,1);
    FP_test = testConfMat(1,2);
    TP_test = testConfMat(2,2);
    
    test_accuracy = (TP_test + TN_test) / (TN_test + FN_test + FP_test + TP_test + eps) * 100;
    test_precision = TP_test / (TP_test + FP_test + eps) * 100;
    test_recall = TP_test / (TP_test + FN_test + eps) * 100;
    
    % TRAIN Metrics
    TN_train = trainConfMat(1,1);
    FN_train = trainConfMat(2,1);
    FP_train = trainConfMat(1,2);
    TP_train = trainConfMat(2,2);
    
    train_accuracy = (TP_train + TN_train) / (TN_train + FN_train + FP_train + TP_train + eps) * 100;
    train_precision = TP_train / (TP_train + FP_train + eps) * 100;
    train_recall = TP_train / (TP_train + FN_train + eps) * 100;
    
    % Store metrics
    test_accuracies(k) = test_accuracy;
    test_precisions(k) = test_precision;
    test_recalls(k) = test_recall;
    train_accuracies(k) = train_accuracy;
    train_precisions(k) = train_precision;
    train_recalls(k) = train_recall;
    
    % Print fold results
    fprintf('Testing Results:\n');
    fprintf('  Accuracy: %.2f%%\n', test_accuracy);
    fprintf('  Precision: %.2f%%\n', test_precision);
    fprintf('  Recall: %.2f%%\n', test_recall);
    fprintf('Training Results:\n');
    fprintf('  Accuracy: %.2f%%\n', train_accuracy);
    fprintf('  Precision: %.2f%%\n', train_precision);
    fprintf('  Recall: %.2f%%\n\n', train_recall);
end

%% Final Report
fprintf('\nFinal Results (Avg over %d folds)\n', K);
fprintf('----------------------------------\n');
fprintf('Testing Metrics:\n');
fprintf('Average Accuracy: %.2f%%\n', mean(test_accuracies));
fprintf('Average Precision: %.2f%%\n', mean(test_precisions));
fprintf('Average Recall: %.2f%%\n', mean(test_recalls));
fprintf('Training Metrics:\n');
fprintf('Average Accuracy: %.2f%%\n', mean(train_accuracies));
fprintf('Average Precision: %.2f%%\n', mean(train_precisions));
fprintf('Average Recall: %.2f%%\n', mean(train_recalls));

%% Combined confusion matrices
confusionchart(testCombinedConfMat, {'No Diabetes', 'Diabetes'});
title('Test Set: Combined Confusion Matrix (Across All Folds)');

%% Train final model on all data for visualization
finalDT = fitctree(features_norm, labels, ...
    'MinLeafSize', 8, ...
    'MaxNumSplits', 15, ...
    'SplitCriterion', 'gdi');

% Visualize the optimized decision tree
figure;
view(finalDT, 'Mode', 'graph');
title('Optimized Decision Tree Visualization');
