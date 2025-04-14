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

% Create clean dataset - use all features for now
clean_predictors = predictors(complete_cases, :);
clean_target = target(complete_cases);

%% PCA Feature Extraction
% Select the same columns as in your PCA script
X = clean_predictors(:, [2, 3, 5, 6, 7]); % Glucose, BP, Insulin, BMI, DPF
Y = clean_target;
PCA_VarName = {'Glucose'; 'Blood Pressure'; 'Insulin'; 'BMI'; 'DPF'};

% Center and scale X
[N, K] = size(X);
X_ctr = zeros(N, K);
X_CS = zeros(N, K);
for k = 1:K
    X_ctr(:, k) = X(:, k) - mean(X(:, k));
    X_CS(:, k) = X_ctr(:, k)/std(X_ctr(:, k));
end

% Run PCA - 3 component model (using MATLAB's built-in PCA function for simplicity)
[coeff, score, latent, ~, explained] = pca(X_CS);
T = score(:, 1:3); % Extract first 3 PCA components

% Show variance explained
fprintf('Variance explained by 3 components: %.2f%%\n', sum(explained(1:3)));

% Create new variable names for PCA components
PCA_CompNames = {'PC1'; 'PC2'; 'PC3'};

% Calculate the number of samples
NumSamples = size(T, 1);

% Random seed for reproducibility - use same seed as other iterations
rng(999);

%% Cross-validation Setup
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
    xTrain = T(trainIdx, :);
    yTrain = Y(trainIdx);
    xTest = T(testIdx, :);
    yTest = Y(testIdx);
    
    % Create decision tree with optimized hyperparameters
    dt = fitctree(xTrain, yTrain, ...
        'MinLeafSize', 8, ...        % Prevent overfitting with larger leaf size
        'MaxNumSplits', 15, ...      % Control tree complexity
        'SplitCriterion', 'gdi', ... % Gini's diversity index
        'PredictorNames', PCA_CompNames);
    
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
    fprintf('Accuracy: %.2f%%\n', test_accuracy);
    fprintf('Precision: %.2f%%\n', test_precision);
    fprintf('Recall: %.2f%%\n', test_recall);
    fprintf('Training Results:\n');
    fprintf('Accuracy: %.2f%%\n', train_accuracy);
    fprintf('Precision: %.2f%%\n', train_precision);
    fprintf('Recall: %.2f%%\n\n', train_recall);
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
finalDT = fitctree(T, Y, ...
    'MinLeafSize', 8, ...
    'MaxNumSplits', 15, ...
    'SplitCriterion', 'gdi', ...
    'PredictorNames', PCA_CompNames);

% Visualize the optimized decision tree
figure;
view(finalDT, 'Mode', 'graph');
title('PCA + Decision Tree Visualization');
