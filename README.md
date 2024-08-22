# **Customer Churn Prediction**

## **Project Overview**

This project is focused on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to churn based on historical data. This information is crucial for businesses to implement strategies to retain their customers and improve overall customer satisfaction.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploration, Cleaning, Validation, and Visualization](#exploration-cleaning-validation-and-visualization)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Prediction Submission](#prediction-submission)
- [File Descriptions](#file-descriptions)
- [Acknowledgements](#acknowledgements)

## **Installation**

To run this project, you need to have Python installed on your system. The following Python packages are also required:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install these packages using pip:

```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
````

# Customer Churn Prediction Project

## Dataset

The dataset contains information about customers, including various features that can be used to predict whether or not a customer will churn. The dataset is split into two files:

- **train.csv**: This file contains the training data, which includes the features and the target variable (`Churn`).
- **test.csv**: This file contains the test data, where predictions will be made.

### Features:
- **CustomerID**: Unique identifier for each customer.
- **Various demographic and service-related features**: The dataset includes a variety of features related to the customer's demographics, account information, and service usage.
- **Churn**: Target variable indicating whether the customer has churned (1) or not (0).

## Exploration, Cleaning, Validation, and Visualization

Before modeling, the data is thoroughly explored, cleaned, validated, and visualized:

### Exploration:
- Summary statistics and missing values are analyzed.
- Initial visualizations are created to understand the distribution of features.

### Cleaning:
- Missing values are handled appropriately.
- Categorical features are encoded using label encoding.

### Validation:
- Class distributions and feature correlations are checked to ensure data integrity.

### Visualization:
- Various plots (e.g., bar plots, box plots, correlation heatmaps) are used to visualize the relationship between features and the target variable (`Churn`).

## Modeling

A `RandomForestClassifier` is used as the primary model in this project. The process includes:

### Data Splitting:
- The data is split into training and validation sets to evaluate model performance.

### Feature Scaling:
- Features are standardized using `StandardScaler` to ensure that they are on the same scale.

### Model Training:
- The model is trained using the training data and then evaluated on the validation data.

### Hyperparameter Tuning (optional):
- Hyperparameters can be tuned to optimize the model's performance.

## Evaluation

The model's performance is evaluated using several metrics:

- **AUC-ROC Score**: The area under the ROC curve is calculated to assess the model's ability to distinguish between classes.
- **Confusion Matrix**: The confusion matrix is used to visualize the model's performance in terms of true positives, true negatives, false positives, and false negatives.

## Prediction Submission

The model generates predictions on the test dataset. These predictions are saved in a CSV file named `prediction_submission.csv`, containing:

- **CustomerID**: The unique identifier of the customer.
- **predicted_probability**: The predicted probability that the customer will churn.

## File Descriptions

- **train.csv**: The training dataset with customer features and the target variable (`Churn`).
- **test.csv**: The test dataset without the target variable.
- **main.py**: The main script for data processing, model training, and prediction.
- **prediction_submission.csv**: The final output file containing predictions on the test dataset.

## Acknowledgements

The dataset used in this project is provided by [Your Data Source]. This project was inspired by the need to improve customer retention strategies through predictive modeling.

