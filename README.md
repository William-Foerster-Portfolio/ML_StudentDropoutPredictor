# Student Dropout Prediction

This project aims to predict student dropout using various machine learning models, including k-Nearest Neighbors (KNN), Decision Trees, Random Forests, and Support Vector Machines (SVM). The project includes data preprocessing, feature selection, model training, and evaluation to identify the most effective model for predicting student dropout.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)

## Overview



## Dataset

The dataset used in this project contains various features related to students, including demographic information, academic performance, and socio-economic indicators. The target variable is `Target`, which represents the dropout status of the students. The dataset is provided in a CSV file named `dropout_data.csv`.

## Dependencies

To run this project, you need to have Python installed along with the necessary libraries. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone https://github.com/will-foerster-portfolio/ML_StudentDropoutPredictor.git
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
## Data Preprocessing

Data preprocessing steps include:

- **Loading the dataset**: The dataset is loaded from the dropout_data.csv file using pandas.
- **Dropping irrelevant features**: Features such as Marital status, Application order, Course, and others that do not contribute to the predictive modeling are removed.
- **Filtering target variable**: Rows where the Target value is 'Enrolled' are removed, focusing only on students who either dropped out or completed.
- **Encoding categorical variables**: Categorical variables are encoded into numerical format using LabelEncoder for compatibility with machine learning models.

## Modeling

### K-Nearest Neighbors (KNN)

The K-Nearest Neighbors (KNN) algorithm is implemented from scratch in this project using a BallTree structure to efficiently find the nearest neighbors. The steps involved in the KNN modeling are:

1. **Training the KNN Model**: The model is trained using the training dataset. The `KNN` class is designed to store training data and perform classification using the BallTree algorithm.
   
2. **Selecting the Optimal K**: Different values of K (the number of nearest neighbors) are tested to find the optimal value that maximizes the model’s accuracy. The model is evaluated on the validation set for each K value.

3. **Predicting and Evaluating**: Once the optimal K is determined, the model is used to make predictions on the validation set. The accuracy, precision, and recall are calculated to evaluate the model's performance.

### Decision Tree Classifier

A Decision Tree Classifier is used to predict student dropout. The steps involved in this process are:

1. **Training the Decision Tree**: The `DecisionTreeClassifier` from `scikit-learn` is used to train the model on the preprocessed training dataset.

2. **Visualizing the Decision Tree**: The trained decision tree is visualized using `Graphviz` and `pydotplus` to understand the decision rules applied by the model. This helps in interpreting the model’s decision-making process.

3. **Evaluating the Model**: The trained model is evaluated using the validation set. The evaluation metrics such as accuracy, precision, and recall are computed to assess the model’s effectiveness.

### Random Forest Classifier

The Random Forest Classifier, an ensemble learning method, is also applied to the dataset:

1. **Training the Random Forest**: A `RandomForestClassifier` from `scikit-learn` is used to train the model on the training dataset. This classifier builds multiple decision trees and merges them together to get a more accurate and stable prediction.

2. **Hyperparameter Tuning**: Hyperparameters such as the number of trees (`n_estimators`) and maximum depth of trees (`max_depth`) are tuned using `RandomizedSearchCV` to find the best combination of parameters for optimal model performance.

3. **Evaluating the Model**: The best model, as determined by hyperparameter tuning, is evaluated using accuracy, precision, and recall metrics on the validation set.

### Support Vector Machine (SVM)

Support Vector Machines (SVM) are also utilized to predict dropout:

1. **Training Different Kernels**: Four different SVM kernels—linear, polynomial, radial basis function (RBF), and sigmoid—are used to train the model. Each kernel can capture different types of relationships within the data.

2. **Evaluating Each Kernel**: Each trained model is evaluated on the validation set using accuracy, precision, and recall to determine the performance of each kernel type.

3. **Comparing Kernel Performance**: The results from each kernel are compared to identify which one provides the best predictive accuracy and overall performance for the dropout prediction task.

## Evaluation

To evaluate the performance of the models, several metrics are used:

- **Accuracy**: The proportion of correctly predicted instances out of the total instances. It is a basic measure of model performance.
  
- **Precision**: The proportion of true positive predictions to the total number of positive predictions made by the model. It indicates the accuracy of the positive predictions.

- **Recall**: The proportion of true positive predictions to the total actual positives in the dataset. It shows the model's ability to identify all the positive cases.

- **Confusion Matrix**: A table used to describe the performance of a classification model by displaying the true positives, true negatives, false positives, and false negatives. It provides a more detailed performance analysis.

Each model is evaluated on the validation set using these metrics to provide a comprehensive view of its performance.

## Results

### K-Nearest Neighbors (KNN)

- **Best Accuracy**: 0.8077
- **Best Precision**: 0.8486
- **Best Recall**: 0.8077

**Figure 3: Accuracy vs. Number of Neighbors** shows that the KNN model is most accurate when `k` is greater than or equal to 11, 12, 23, or 24. The model achieves a balance between accuracy and precision at these values, making it effective for predicting student dropout.

### Decision Tree Classifier

- **Best Accuracy**: 0.8769
- **Best Precision**: 0.8817
- **Best Recall**: 0.8769

**Figure 4: max_depth vs. accuracy** illustrates that the Decision Tree model achieves its best accuracy when the maximum depth (`max_depth`) is set to 9. This indicates that a moderately deep tree provides the best balance between model complexity and generalization ability.

### Random Forest Classifier

- **Best Hyperparameters**: `{'max_depth': 16, 'n_estimators': 430}`
- **Best Accuracy**: 0.8846
- **Best Precision**: 0.8882
- **Best Recall**: 0.8846

The Random Forest Classifier achieves the highest accuracy among the models after hyperparameter tuning. With a maximum depth of 16 and 430 estimators, it balances both precision and recall effectively, as reflected in the confusion matrix and the evaluation metrics.

### Support Vector Machine (SVM)

- **Linear Kernel**:
- **Accuracy**: 0.8692
- **Precision**: 0.8884
- **Recall**: 0.8692

- **Polynomial Kernel**:
- **Accuracy**: 0.7308
- **Precision**: 0.7720
- **Recall**: 0.7308

- **Radial Basis Function (RBF) Kernel**:
- **Accuracy**: 0.5385
- **Precision**: 0.2899
- **Recall**: 0.5385

- **Sigmoid Kernel**:
- **Accuracy**: 0.5077
- **Precision**: 0.5042
- **Recall**: 0.5077

The SVM model with the **Linear Kernel** performs the best among the four kernels tested, with a high accuracy and precision. The **Polynomial Kernel** shows moderate performance, while the **RBF** and **Sigmoid Kernels** perform poorly, indicating they are less suitable for this specific classification task.

### Summary

The table below summarizes the performance of each model based on accuracy, precision, and recall:

| Model                                  | Accuracy | Precision | Recall |
|----------------------------------------|----------|-----------|--------|
| Random Forest Classifier               | 0.8846   | 0.8882    | 0.8846 |
| Decision Tree Classifier               | 0.8769   | 0.8817    | 0.8769 |
| SVM (Linear Kernel)                    | 0.8692   | 0.8884    | 0.8692 |
| K-Nearest Neighbors (K=11, 12, 23, 24) | 0.8077   | 0.8486    | 0.8077 |
| SVM (Polynomial Kernel)                | 0.7308   | 0.7720    | 0.7308 |
| SVM (RBF Kernel)                       | 0.5385   | 0.2899    | 0.5385 |
| SVM (Sigmoid Kernel)                   | 0.5077   | 0.5042    | 0.5077 |

The **Random Forest Classifier** demonstrates the best overall performance with an accuracy of 0.8846, followed closely by the **Decision Tree Classifier** and the **Support Vector Machine (Linear Kernel)** model. The **K-Nearest Neighbors (KNN)** also performs well but is slightly less accurate compared to the Random Forest and Decision Tree, and SVM (Linear Kernal)models. 

Based on these results, the Random Forest Classifier is recommended for predicting student dropout due to its superior accuracy and balanced precision and recall scores.
