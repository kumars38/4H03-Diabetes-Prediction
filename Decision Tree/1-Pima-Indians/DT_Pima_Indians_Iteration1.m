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

% Random training/testing split - use same seed as ANN
rng(999)

% Extract 80% of data for training and remainder for testing
TrainNumSamples = round(0.8 * NumSamples);
TrainRowInds = randperm(NumSamples, TrainNumSamples);
TrainFeatures = features(TrainRowInds, :);
TrainLabels = labels(TrainRowInds);
TestFeatures = features(setdiff(1:NumSamples, TrainRowInds), :);
TestLabels = labels(setdiff(1:NumSamples, TrainRowInds));  
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
TN = dtTest_confusionmat(1,1);
FN = dtTest_confusionmat(2,1);
FP = dtTest_confusionmat(1,2);
TP = dtTest_confusionmat(2,2);

% Calculate metrics
testAccuracy = (TP + TN) / (TN + FN + FP + TP) * 100;
testPrecision = TP / (TP + FP + eps) * 100;
testRecall = TP / (TP + FN + eps) * 100;

% Calculate training metrics
dtTrain_confusionmat = confusionmat(TrainLabels, dtTrainPredLabels);
TN_train = dtTrain_confusionmat(1,1);
FN_train = dtTrain_confusionmat(2,1);
FP_train = dtTrain_confusionmat(1,2);
TP_train = dtTrain_confusionmat(2,2);

trainAccuracy = (TP_train + TN_train) / (TN_train + FN_train + FP_train + TP_train) * 100;
trainPrecision = TP_train / (TP_train + FP_train + eps) * 100;
trainRecall = TP_train / (TP_train + FN_train + eps) * 100;

%% Final Report
fprintf('\nFinal Results Summary\n');
fprintf('-------------------\n');
fprintf('Testing Metrics:\n');
fprintf('Accuracy: %.2f%%\n', testAccuracy);
fprintf('Precision: %.2f%%\n', testPrecision);
fprintf('Recall: %.2f%%\n', testRecall);
fprintf('\nTraining Metrics:\n');
fprintf('Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Precision: %.2f%%\n', trainPrecision);
fprintf('Recall: %.2f%%\n', trainRecall);

% Show the training confusion matrix
figure
confusionchart(TrainLabels, dtTrainPredLabels, 'Title', 'Decision Tree Classifier Training Confusion Matrix');

% Visualize results with confusion matrix
figure
confusionchart(TestLabels, dtTestPredLabels, 'Title', 'Decision Tree Classifier Testing Confusion Matrix');

% Show the training confusion matrix
figure
confusionchart(TrainLabels, dtTrainPredLabels, 'Title', 'Decision Tree Classifier Training Confusion Matrix');

% Visualize the decision tree
figure
view(dt, 'Mode', 'graph');
title('Decision Tree Visualization');