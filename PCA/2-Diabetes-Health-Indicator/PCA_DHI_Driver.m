% CHEMENG 4H03 Project
% Group 5
% Written by Steven Chui on 2025-03-22

% Diabetes Health Indicator Dataset
clear variables; close all; clc;

%% Import dataset
data = readmatrix("diabetes_health_indicators.csv");
X = data(:,[5,15,16,17,20,21,22]);
Y = data(:,1);
VarName = {'BMI';'GenHlth';'MentHlth';'PhysHlth';
    'Age';'Education';'Income'};
[N,K] = size(X);
[~,M] = size(Y);

% Center and scale X
X_ctr = zeros(N,K);
X_CS = zeros(N,K);
for k = 1:K
    X_ctr(:,k) = X(:,k) - mean(X(:,k));
    X_CS(:,k) = X_ctr(:,k)/std(X_ctr(:,k));
end %for

%% Scatterplot matrix to indentify any covariance within X
F1 = figure();
[~,ax] = plotmatrix(X_CS);

for i = 1:K
    xlabel(ax(K,i), VarName(i))
    ylabel(ax(i,1), VarName(i))
end % for

%% PCA by nipals

% PCA with cross validation
G = 2;      % number of groups to split the data into
A_max = 5;  % maximum number of principle components needed to be fitted

% Assign a random group value to each row in X_CS
rand_seed = randi([1,G],G,1);
gp_indx = repmat(rand_seed,N/G,1);

% Placeholders
R2_vec = zeros(1,A_max);
Q2_vec = zeros(1,A_max);


for a = 1:A_max
    PRESS = 0;

    for g = 1:G
        train_indx = 1;
        test_indx = 1;
        X_train = zeros(N-N/G,K);
        X_test = zeros(N/G,K);
        % Assign training and testing set

        for i = 1:N
            if gp_indx(i) == g
                X_test(test_indx,:) = X_CS(i,:);
                test_indx = test_indx + 1;
            else
                X_train(train_indx,:) = X_CS(i,:);
                train_indx = train_indx + 1;
            end %if
        end % for i

        % fit 'a' component onto the training data
        [~,P,~] = nipalspca(X_train,a);

        % Project testing data onto the model
        T_new = X_test*P;
        X_new_hat = T_new*P';

        % Calculate PRESS and R2_testing (Q2)
        E_new = X_test - X_new_hat;
        PRESS = PRESS + sum(sum(E_new.^2));
    end % for g

    Q2_vec(a) = 1 - PRESS/sum(sum(X_CS.^2));

    % Fit a full PCA model with 'a' component
    % with X_CS
    [~,~,R2] = nipalspca(X_CS,a);
    R2_vec(a) = R2(a);

end % for a

disp('R2_vec')
disp(R2_vec)
disp('Q2_vec')
disp(Q2_vec)

% Both R2 and Q2 are low (at about 0.2) - PCA is not quite suitable for
% this dataset

% % %% Plots
% % % Let's build a 2 component model regardless
% % [T,P,R2] = nipalspca(X,2);
% % 
% % % Score plots functions below created by Dr. Jake Nease
% % scoreplot(T(:,1), T(:,2));
% % score_loading_plot(T(:,1), T(:,2),P(:,1),P(:,2), VarName);