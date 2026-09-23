# Loan Default Prediction | [Kaggle Link](https://www.kaggle.com/competitions/dan-it-data-science-step-project-1/overview) | [View Code](https://github.com/Matsalak-Viktoria/Default-Probability-Prediction/blob/main/Default_Probability_Prediction.ipynb)

## Overview
This project explores the implementation and evaluation of a machine learning pipeline for loan default prediction using customer credit and loan data.

The main goal of the project is to build a classification model for predicting the probability of loan default using exploratory data analysis, data preprocessing, feature engineering, feature selection, and hyperparameter tuning with Optuna.

The project focuses on the following prediction task:
- Loan Default Prediction - Predicting the probability that a customer will default on a loan based on demographic, financial, and credit history information.

## Objectives
The main objectives of this project are:
- Perform Exploratory Data Analysis (EDA) to understand feature distributions and relationships with the target variable.
- Prepare the data for modeling through data preprocessing, feature engineering, and feature selection.
- Build a LightGBM classification model for loan default prediction and optimize its hyperparameters using Optuna.
- Evaluate the model's predictive performance using ROC AUC.
- Interpret the model's results using feature importance and SHAP.

## Dataset
- **UniqueID**:	Unique identifier for each customer record.
- **disbursed_amount**:	Amount of the loan disbursed to the customer.
- **asset_cost**:	Cost of the asset.
- **ltv**: Loan-to-value of the asset.
- **branch_id**: Identifier of the branch where the loan was disbursed.
- **supplier_id**: Identifier of the supplier.
- **manufacturer_id**: Identifier of the manufacturer.
- **Current_pincode_ID**: Identifier of the customer's current pincode.
- **Date.of.Birth**: Customer's date of birth.
- **Employment.Type**: Customer's employment type (Salaried/Self Employed).
- **DisbursalDate**: Date of loan disbursement.
- **State_ID**:	Identifier of the state.
- **Employee_code_ID**:	Identifier of the employee associated with the loan disbursement.
- **MobileNo_Avl_Flag**: Indicates whether a mobile phone number was provided by the customer (0 = No, 1 = Yes).
- **Aadhar_flag**: Indicates whether Aadhaar was provided by the customer (0 = No, 1 = Yes).
- **PAN_flag**:	Indicates whether PAN was provided by the customer (0 = No, 1 = Yes).
- **VoterID_flag**:	Indicates whether Voter ID was provided by the customer (0 = No, 1 = Yes).
- **Driving_flag**:	Indicates whether a driving license was provided by the customer (0 = No, 1 = Yes).
- **Passport_flag**: Indicates whether a passport was provided by the customer (0 = No, 1 = Yes).
- **PERFORM_CNS.SCORE**: Customer's credit bureau score.
- **PERFORM_CNS.SCORE.DESCRIPTION**: Textual interpretation of the customer's credit bureau score.
- **PRI.NO.OF.ACCTS**: Number of primary credit accounts.
- **PRI.ACTIVE.ACCTS**:	Number of active primary credit accounts.
- **PRI.OVERDUE.ACCTS**: Number of overdue primary credit accounts.
- **PRI.CURRENT.BALANCE**: Current balance of the customer's primary credit accounts.
- **PRI.SANCTIONED.AMOUNT**: Sanctioned amount of the customer's primary credit accounts.
- **PRI.DISBURSED.AMOUNT**:	Disbursed amount of the customer's primary credit accounts.
- **SEC.NO.OF.ACCTS**: Number of secondary credit accounts.
- **SEC.ACTIVE.ACCTS**:	Number of active secondary credit accounts.
- **SEC.OVERDUE.ACCTS**: Number of overdue secondary credit accounts.
- **SEC.CURRENT.BALANCE**: Current balance of the customer's secondary credit accounts.
- **SEC.SANCTIONED.AMOUNT**: Sanctioned amount of the customer's secondary credit accounts.
- **SEC.DISBURSED.AMOUNT**:	Disbursed amount of the customer's secondary credit accounts.
- **PRIMARY.INSTAL.AMT**:	Total installment amount associated with primary credit accounts.
- **SEC.INSTAL.AMT**:	Total installment amount associated with secondary credit accounts.
- **NEW.ACCTS.IN.LAST.SIX.MONTHS**:	Number of new credit accounts opened in the last six months.
- **DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS**:	Number of delinquent credit accounts in the last six months.
- **AVERAGE.ACCT.AGE**:	Average age of the customer's credit accounts.
- **CREDIT.HISTORY.LENGTH**:	Length of the customer's credit history.
- **NO.OF_INQUIRIES**:	Number of credit bureau inquiries associated with the customer.
- **loan_default**:	Target variable indicating whether the loan defaulted (0 = No Default, 1 = Default).

## Workflow

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Methods
### Data Preprocessing
**For Outlier Detection**:
- Missing value imputation (Median)
- Feature scaling (StandardScaler)

**For Model Training**:

**Numerical features**:
- Missing value imputation (Median)
- Feature scaling (StandardScaler)

**Categorical features**:
- Missing value imputation (Most Frequent)
- One-Hot Encoding

### Outlier Detection
- Isolation Forest

### Machine Learning Models
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)

**Hyperparameters optimized**:
- Logistic Regression: Regularization strength (C)
- Decision Tree: Maximum tree depth (max_depth)
- K-Nearest Neighbors (KNN): Number of neighbors (n_neighbors)

### Validation Strategy
**Train/Test Split + GridSearchCV**:
- Train/Test split for final model evaluation
- GridSearchCV with 5-Fold Cross-Validation for hyperparameter optimization

### Evaluation Metrics

## Results
