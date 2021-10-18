"""
A file for altering constants and parameters
author: Ben E
Date: 2021-10-18
"""
# name of target
TARGET_COL = 'Churn'

# names of file paths
FP_DATA = './data/'
FP_EDA = './images/eda/'
FP_MODELS = './models/'
FP_RESULTS = './images/results/'
FP_LOGS = './logs/'

# names of files
FILE_NAME = FP_DATA + r"bank_data.csv"
MODEL_NAME_1 = FP_MODELS + 'rfc_model.pkl'
PLOT_MODEL_1_RESULTS = FP_RESULTS + 'rf_model_results.png'
MODEL_NAME_2 = FP_MODELS + 'logistic_model.pkl'
PLOT_MODEL_2_RESULTS = FP_RESULTS + 'lr_model_results.png'
PLOT_ROC = FP_RESULTS + 'lr_vs_rf_roc_plot.png'
PLOT_FEAT_IMP = FP_RESULTS + 'feature_importance_values.png'
PLOT_SHAP = FP_RESULTS + 'shap_values.png'
LOG_FILE = FP_LOGS + 'churn_library.log'

# variable adjustment names
VAR_1 = 'Attrition_Flag'
FIELD_1 = 'Existing Customer'

# variable list for plotting
dict_vars_plot = {
        'Churn':'hist',
        'Customer_Age':'hist',
        'Marital_Status':'value_counts',
        'Total_Trans_Ct':'distplot'
}

# feature list for encoding
lst_features_encode = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

# variables used for modeling
lst_keep_features = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

# parameters for RF grid search
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4 , 5, 100],
    'criterion': ['gini', 'entropy']
}

