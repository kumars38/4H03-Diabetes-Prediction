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

%% Pre-Processing

% Identify columns where 0s represent missing values
columns_to_check = [2, 3, 4, 5, 6]; % Glucose, BP, Skin, Insulin, BMI
column_names = {'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI'};

% Create a logical array to identify rows with missing values
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

% Create clean dataset with only complete cases
features = predictors(complete_cases, :); % Changed variable name for consistency
labels = target(complete_cases);          % Changed variable name for consistency

% Calculate the number of samples
NumSamples = size(features, 1);          % Added calculation of NumSamples

% Random training/testing split
rng("default")

% Extract 80% of data for training and remainder for testing
TrainNumSamples = round(0.8 * NumSamples);
TrainRowInds = randperm(NumSamples, TrainNumSamples);
TrainFeatures = features(TrainRowInds, :);
TrainLabels = labels(TrainRowInds);
TestFeatures = features(setdiff(1:NumSamples, TrainRowInds), :);
TestLabels = labels(setdiff(1:NumSamples, TrainRowInds));  % Removed extra dimension
TestNumSamples = length(TestLabels);


%-----------------------------Decision Trees------------------------------%

% Create decision trees classifier
dt = fitctree(TrainFeatures, TrainLabels);

% Predict labels for the test set
dtTestPredLabels = predict(dt, TestFeatures);

% Predict labels for the train set
dtTrainPredLabels = predict(dt, TrainFeatures);

% Calculate testing accuracy
dtTest_confusionmat = confusionmat(TestLabels, dtTestPredLabels);
dtTestAccuracy = sum(diag(dtTest_confusionmat))/sum(dtTest_confusionmat(:));
fprintf('Decision Tree Testing Accuracy: %.2f%%\n', dtTestAccuracy*100);

% Visualize results with confusion matrix
figure
confusionchart(TestLabels, dtTestPredLabels, 'Title', 'Decision Tree Classifier Testing Confusion Matrix');

% Calculate training accuracy
dtTrain_confusionmat = confusionmat(TrainLabels, dtTrainPredLabels);
dtTrainAccuracy = sum(diag(dtTrain_confusionmat))/sum(dtTrain_confusionmat(:));
fprintf('Decision Tree Training Accuracy: %.2f%%\n', dtTrainAccuracy*100);

% Show the training confusion matrix
figure
confusionchart(TrainLabels, dtTrainPredLabels, 'Title', 'Decision Tree Classifier Training Confusion Matrix');

% Visualize the decision tree
figure
view(dt, 'Mode', 'graph');
title('Decision Tree Visualization');

% Feature importance analysis
importance = predictorImportance(dt);
figure
bar(importance);
title('Feature Importance in Decision Tree Model');
xlabel('Feature Index');
ylabel('Importance Score');
xticks(1:length(VarName));
xticklabels(VarName);
xtickangle(45);