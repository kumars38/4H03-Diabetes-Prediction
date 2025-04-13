% Sections modified from MATLAB's ANN -> generated script
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

%% Pre-Processing - Same as in PCA script
% Identify columns with missing values
columns_to_check = [2, 3, 4, 5, 6]; % Glucose, BP, Skin, Insulin, BMI
column_names = {'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI'};

% Identify rows with missing values
has_missing = false(size(predictors, 1), 1);
for i = 1:length(columns_to_check)
    col = columns_to_check(i);
    zero_indices = predictors(:, col) == 0;
    has_missing = has_missing | zero_indices;
    fprintf('Column %s has %d missing values (zeros)\n', column_names{i}, sum(zero_indices));
end

% Remove rows with missing values
complete_cases = ~has_missing;
fprintf('Removing %d rows with missing data (%.1f%% of the dataset)\n', ...
        sum(has_missing), 100*sum(has_missing)/length(has_missing));
fprintf('Keeping %d complete cases (%.1f%% of the dataset)\n', ...
        sum(complete_cases), 100*sum(complete_cases)/length(complete_cases));

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

%% Decision Tree with PCA Components
% Create new variable names for PCA components
PCA_CompNames = {'PC1'; 'PC2'; 'PC3'};

% Calculate the number of samples
NumSamples = size(T, 1);

% Random training/testing split
rng("default");

% Extract 80% of data for training and remainder for testing
TrainNumSamples = round(0.8 * NumSamples);
TrainRowInds = randperm(NumSamples, TrainNumSamples);
TrainFeatures_PCA = T(TrainRowInds, :);
TrainLabels = Y(TrainRowInds);
TestFeatures_PCA = T(setdiff(1:NumSamples, TrainRowInds), :);
TestLabels = Y(setdiff(1:NumSamples, TrainRowInds));
TestNumSamples = length(TestLabels);

%-----------------------------Decision Trees with PCA------------------------------%

% Create decision trees classifier with PCA components
dt_pca = fitctree(TrainFeatures_PCA, TrainLabels, 'PredictorNames', PCA_CompNames);

% Predict labels for the test set
dtTestPredLabels_pca = predict(dt_pca, TestFeatures_PCA);

% Predict labels for the train set
dtTrainPredLabels_pca = predict(dt_pca, TrainFeatures_PCA);

% Calculate testing accuracy
dtTest_confusionmat_pca = confusionmat(TestLabels, dtTestPredLabels_pca);
dtTestAccuracy_pca = sum(diag(dtTest_confusionmat_pca))/sum(dtTest_confusionmat_pca(:));
fprintf('PCA + Decision Tree Testing Accuracy: %.2f%%\n', dtTestAccuracy_pca*100);

% Calculate training accuracy
dtTrain_confusionmat_pca = confusionmat(TrainLabels, dtTrainPredLabels_pca);
dtTrainAccuracy_pca = sum(diag(dtTrain_confusionmat_pca))/sum(dtTrain_confusionmat_pca(:));
fprintf('PCA + Decision Tree Training Accuracy: %.2f%%\n', dtTrainAccuracy_pca*100);

% Visualize results with confusion matrix
figure
confusionchart(TestLabels, dtTestPredLabels_pca, 'Title', 'PCA + Decision Tree Testing Confusion Matrix');

% Visualize the decision tree
figure
view(dt_pca, 'Mode', 'graph');
title('PCA + Decision Tree Visualization');

% Feature importance analysis for PCA components
importance_pca = predictorImportance(dt_pca);
figure
bar(importance_pca);
title('PCA Component Importance in Decision Tree Model');
xlabel('PCA Component');
ylabel('Importance Score');
xticks(1:length(PCA_CompNames));
xticklabels(PCA_CompNames);

%% For comparison - also run decision tree on original features
% Create training/testing datasets with original features
TrainFeatures_orig = clean_predictors(TrainRowInds, :);
TestFeatures_orig = clean_predictors(setdiff(1:NumSamples, TrainRowInds), :);

% Create decision trees classifier with original features
dt_orig = fitctree(TrainFeatures_orig, TrainLabels);

% Predict labels and calculate accuracy
dtTestPredLabels_orig = predict(dt_orig, TestFeatures_orig);
dtTest_confusionmat_orig = confusionmat(TestLabels, dtTestPredLabels_orig);
dtTestAccuracy_orig = sum(diag(dtTest_confusionmat_orig))/sum(dtTest_confusionmat_orig(:));
fprintf('Original Features Decision Tree Testing Accuracy: %.2f%%\n', dtTestAccuracy_orig*100);

% Compare methods
fprintf('\n----- COMPARISON -----\n');
fprintf('Original Features DT Accuracy: %.2f%%\n', dtTestAccuracy_orig*100);
fprintf('PCA Components DT Accuracy: %.2f%%\n', dtTestAccuracy_pca*100);