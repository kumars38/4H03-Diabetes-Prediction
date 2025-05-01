# 4H03-Diabetes-Prediction
**Contributors**: Elsie Igho, Steven Chui, Samarth Kumar, James Park

We created a diabetes screening tool based on modeling public data involving health metrics from diabetic/non-diabetic patients.

**Datasets**:
1) Pima-Indians: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data
2) Diabetes-Health-Indicators: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data
3) Diabetes-Dataset: https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset
4) Mendeley-Diabetes-Dataset: https://data.mendeley.com/datasets/wj9rwkp9c2/1

**Credits to**: Alexandre D'Souza (Chemical Engineering, McMaster University) for providing the starter code.

Please use the links below to access the models and their code.

## PCA
PCA models for the Pima-Indians dataset.

[5-component model on 8 variables (i.e.,full dataset, no NAN)](PCA/1-Pima-Indians/PCA_Pima_Indians_Driver.m)

[3-component model on 5 selected variables](PCA/1-Pima-Indians/PCA_Pima_Indians_Driver_Full.m)

## ANN
ANN models for the Pima-Indians dataset.

[ANN using raw data, trainlm, MSE](ANN/1-Pima-Indians/ANN_Pima_Indians_Driver_Iteration_1.m)

[ANN using normalization, trainscg, binary cross-entropy](ANN/1-Pima-Indians/ANN_Pima_Indians_Driver_Iteration_2.m)

[ANN using PCA-reduced inputs](ANN/1-Pima-Indians/ANN_Pima_Indians_Driver_Iteration_3.m)

## Decision Tree
Decision tree models for the Pima-Indians dataset.

[DT iteration 1](Decision%20Tree/1-Pima-Indians/DT_Pima_Indians_Iteration1.m)

[DT iteration 2](Decision%20Tree/1-Pima-Indians/DT_Pima_Indians_Iteration2.m)

[DT using PCA-reduced inputs](Decision%20Tree/1-Pima-Indians/DT_Pima_Indians_Iteration3.m)

