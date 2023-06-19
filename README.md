# Classification-Project-from-Sony-Research

This repository contains the solution for a data project from Sony Research. The goal of the project is to build a predictive model to identify customers who are likely to churn. The project involves exploratory analysis, data preprocessing, model training, and evaluation.

## Table of Contents

1. [Assignment](#assignment)
2. [Exploratory Analysis and Extract Insights](#exploratory-analysis-and-extract-insights)
3. [Train/Test Split](#traintest-split)
4. [Predictive Model](#predictive-model)
5. [Metrics](#metrics)
6. [Model Results](#model-results)
   - [Classical Machine Learning Models](#classical-machine-learning-models)
   - [Deep Learning Model](#deep-learning-model)
7. [Deployment Issues](#deployment-issues)

## Assignment

The assignment involves performing exploratory analysis on the dataset, splitting the data into train and test sets, building a predictive model for churn prediction, establishing metrics for model evaluation, and discussing potential issues with model deployment.

## Exploratory Analysis and Extract Insights

In this section, exploratory analysis is performed on the dataset to gain insights. Various Python libraries such as pandas, seaborn, numpy, and matplotlib.pyplot are imported. The dataset is read using the pandas library, and basic statistics and visualizations are generated to understand the data.

## Train/Test Split

The dataset is split into train and test sets using the train_test_split() function from the sklearn library. The split is done with an 80%-20% ratio, and the stratify option is used to maintain the same class distribution in both sets.

## Predictive Model

In this section, a predictive model is built to predict customer churn. The RandomForestClassifier algorithm is chosen for the task. The features and target variable are separated, and the data is standardized using StandardScaler. The RandomForestClassifier is then trained on the training data.

## Metrics

Metrics are established to evaluate the performance of the predictive model. The f1_score and accuracy_score metrics from the sklearn library are used to assess the model's performance.

## Model Results

The results of the predictive model are analyzed and discussed. The feature importances are calculated using the RandomForestClassifier, and a bar plot is generated to visualize the importance of each feature in predicting churn.

### Classical Machine Learning Models

Various classical machine learning models are imported from the sklearn library, including MLPClassifier, KNeighborsClassifier, SVC, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, GaussianNB, QuadraticDiscriminantAnalysis, XGBClassifier, and LGBMClassifier. These models can be used as alternatives to the RandomForestClassifier for churn prediction.

### Deep Learning Model

A deep learning model is built using the Keras library. The Sequential model is used, and Dense layers are added to construct the neural network. The model can be trained on the data for churn prediction.

## Deployment Issues

Potential issues related to deploying the predictive model into production are discussed. This includes considerations such as model performance, data consistency, model updates, and scalability.

Please note that the code provided in the README is a summary of the solution and may require additional setup or modifications to run successfully.
Please let me know if there's anything else I can help you with!






