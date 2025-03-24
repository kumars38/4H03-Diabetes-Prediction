clc;
clear;
close all;

% Read in data and extract features and labels
data = readtable("diabetes.csv");
data = table2array(data);
[NumSamples, NumVars] = size(data);

% Extract features and labels
features = data(:, 1:end-1); 
labels = data(:, end);   

% Replace zeros with the mean
columnsToClean = [2, 3, 4, 5, 6];
for col = columnsToClean
    zeroIndices = features(:, col) == 0;
    colMean = mean(features(~zeroIndices, col));
    features(zeroIndices, col) = colMean;
end

% Random training/testing split
rng("default")

% Extract 80% of data for training and remainder for testing
TrainNumSamples = round(0.8 * NumSamples);
TrainRowInds = randperm(NumSamples, TrainNumSamples);
TrainFeatures = features(TrainRowInds, :);
TrainLabels = labels(TrainRowInds);
TestFeatures = features(setdiff(1:NumSamples, TrainRowInds), :);
TestLabels = labels(setdiff(1:NumSamples, TrainRowInds), :);
TestNumSamples = length(TestLabels);

%-----------------------------Naive Bayes---------------------------------%

% Create naive Bayes classifier
nb = fitcnb(TrainFeatures, TrainLabels);

% Predict labels for the test set
nbTestPredLabels = predict(nb, TestFeatures);

% Predict labels for the train set
nbTrainPredLabels = predict(nb, TrainFeatures);

% Calculate testing accuracy
nbTest_confusionmat = confusionmat(TestLabels, nbTestPredLabels);
nbTestAccuracy = sum(diag(nbTest_confusionmat))/sum(nbTest_confusionmat(:));
fprintf('Naive Bayes Testing Accuracy: %.2f%%\n', nbTestAccuracy*100);

% Show the testing confusion matrix
figure
confusionchart(TestLabels, nbTestPredLabels, 'Title', 'Naive Bayes Classifier Testing Confusion Matrix');

% Calculate training accuracy
nbTrain_confusionmat = confusionmat(TrainLabels, nbTrainPredLabels);
nbTrainAccuracy = sum(diag(nbTrain_confusionmat))/sum(nbTrain_confusionmat(:));
fprintf('Naive Bayes Training Accuracy: %.2f%%\n', nbTrainAccuracy*100);

% Show the training confusion matrix
figure
confusionchart(TrainLabels, nbTrainPredLabels, 'Title', 'Naive Bayes Classifier Training Confusion Matrix');


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

%----------------------------Neural Networks------------------------------%

% Format for neural network - binary classification
% Convert labels to one-hot encoding format
TrainLabelsMat = zeros(TrainNumSamples, 2);
for i = 1:TrainNumSamples
    TrainLabelsMat(i, TrainLabels(i)+1) = 1; % +1 because MATLAB is 1-indexed
end

TestLabelsMat = zeros(TestNumSamples, 2);
for i = 1:TestNumSamples
    TestLabelsMat(i, TestLabels(i)+1) = 1;
end

% Transpose for neural network input format
TrainFeaturesNN = TrainFeatures';
TrainLabelsMatNN = TrainLabelsMat';
TestFeaturesNN = TestFeatures';
TestLabelsMatNN = TestLabelsMat';

% Create neural network classifier
net = patternnet(10); % 10 neurons in hidden layer
net.trainParam.showWindow = true;
[net, tr] = train(net, TrainFeaturesNN, TrainLabelsMatNN);

% Predict labels for the test set
nnTestPredLabelsMat = net(TestFeaturesNN);

% Predict labels for the train set
nnTrainPredLabelsMat = net(TrainFeaturesNN);

% Convert predictions back to class labels (0 or 1)
[~, nnTestPredLabelsInd] = max(nnTestPredLabelsMat);
nnTestPredLabels = nnTestPredLabelsInd - 1; % Subtract 1 to get 0/1 labels

[~, nnTrainPredLabelsInd] = max(nnTrainPredLabelsMat);
nnTrainPredLabels = nnTrainPredLabelsInd - 1;

% Calculate testing accuracy
nnTest_confusionmat = confusionmat(TestLabels, nnTestPredLabels');
nnTestAccuracy = sum(diag(nnTest_confusionmat))/sum(nnTest_confusionmat(:));
fprintf('Neural Network Testing Accuracy: %.2f%%\n', nnTestAccuracy*100);

% Visualize results with confusion matrix
figure
confusionchart(TestLabels, nnTestPredLabels', 'Title', 'Neural Network Classifier Testing Confusion Matrix');

% Calculate training accuracy
nnTrain_confusionmat = confusionmat(TrainLabels, nnTrainPredLabels');
nnTrainAccuracy = sum(diag(nnTrain_confusionmat))/sum(nnTrain_confusionmat(:));
fprintf('Neural Network Training Accuracy: %.2f%%\n', nnTrainAccuracy*100);

% Visualize results with confusion matrix
figure
confusionchart(TrainLabels, nnTrainPredLabels', 'Title', 'Neural Network Classifier Training Confusion Matrix');