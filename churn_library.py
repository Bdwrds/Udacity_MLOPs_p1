"""
Churn prediction DS process - Functions & Solution

author: Ben E
date: 2021-10-13
"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import shap

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_load = pd.read_csv(pth)
    return df_load

def perform_eda(df_eda):
    '''
    perform eda on df and save figures to images folder
    input:
            df_eda: pandas dataframe

    output:
            None
    '''
    eda_fp = './images/eda/'
    # Dist of Churn
    plt.figure()  # figsize=(20,10))
    df_eda['Churn'].hist()
    plt.savefig(eda_fp + 'eda_churn_dist.png')
    print("saved eda_churn_dist")
    # Dist of Customer_Age
    plt.figure()  # figsize=(20,10))
    df_eda['Customer_Age'].hist()
    plt.savefig(eda_fp + 'eda_age_dist.png')
    print("saved eda_age_dist")
    # Perc of Marital_Status
    plt.figure()  # figsize=(20,10))
    df_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(eda_fp + 'eda_marital_status_perc.png')
    print("saved eda_marital_status_perc")
    # Dist of Total_Trans_Ct
    plt.figure()  # figsize=(20,10))
    sns.distplot(df_eda['Total_Trans_Ct'])
    plt.savefig(eda_fp + 'eda_total_trans_ct_dist.png')
    print("saved eda_total_trans_ct_dist")
    # correlation plot
    plt.figure()  # figsize=(20,10))
    sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(eda_fp + 'eda_heatmap.png')
    print("saved eda_heatmap")

def encoder_helper(df_encode, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_encode: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                [optional argument that could be used for naming variables or index y column]

    output:
            df_encode: pandas dataframe with new columns for
    '''    
    for category in category_lst:
        if df_encode[category].dtype == np.object:
            category_churn = category + '_Churn'
            var_lst = []
            cat_groups = df_encode.groupby(category).mean()[response]

            for val in df_encode[category]:
                var_lst.append(cat_groups.loc[val])

            df_encode[category_churn] = var_lst
        else:
            print("Check category {category} is categorical")
    return df_encode

def perform_feature_engineering(df_eng, keep_cols, response):
    '''
    input:
              df_eng: pandas dataframe
              response: string of response name
                  [optional argument that could be used for naming variables or index y column]

    output:
              feat_train: X training data
              feat_test: X testing data
              target_train: y training data
              target_test: y testing data
    '''
    print('\nFeature engineering step..')
    features = df_eng[keep_cols]
    target = df_eng[response]
    feat_train, feat_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42)
    return feat_train, feat_test, target_train, target_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            output_pth: file path used to save images to

    output:
             None
    '''
    # save image of RF model results
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(output_pth + 'rf_model_results.png')
    print("saved rf_model_results")
    plt.close()

    # save image of LR model results
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(output_pth + 'lr_model_results.png')
    print("saved lr_model_results")
    plt.close()


def shap_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the shap values in pth
    input:
            model: model object containing shap values
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    print('\nFeature importance step..')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    plt.savefig(output_pth + 'shap_values.png')
    print("saved shap_values")


def model_feat_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the model feature importance values in pth
    input:
            model: model object containing shap values
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure()  # figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + 'feature_importance_values.png')
    print("saved feature_importance_values")


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    shap_importance_plot(model, x_data, output_pth)
    model_feat_importance_plot(model, x_data, output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    print("\nTraining models...")
    # define filepaths
    results_image_fp = './images/results/'
    results_model_fp = './models/'

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200],  # , 500],
        'max_features': ['auto'],  # , 'sqrt'],
        'max_depth': [4],  # ,5,100],
        'criterion': ['gini']  # , 'entropy']
    }

    print('\nTraining RF..')
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    print('\nTraining LR..')
    lrc.fit(x_train, y_train)

    # plots
    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(results_image_fp + 'lr_vs_rf_roc_plot.png')
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, results_model_fp + 'rfc_model.pkl')
    joblib.dump(lrc, results_model_fp + 'logistic_model.pkl')
    print("\nTraining process complete...")

    print('\nGenerating report..')
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    # save classification reports on train/test data
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        results_image_fp)


# run if not imported
if __name__ == "__main__":
    FP_DATA = './data/'
    FP_EDA = './images/eda/'
    FP_MODELS = './models/'
    FP_RESULTS = './images/results/'

    df = import_data(pth=FP_DATA + r"bank_data.csv")
    target_col = 'Churn'
    df[target_col] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    
    feature_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    # perform eda
    perform_eda(df, target_col)
    df_encoded = encoder_helper(df, feature_lst, target_col)
    keep_vars = [
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
    feature_train, feature_test, targets_train, targets_test = perform_feature_engineering(
        df_encoded, keep_vars, target_col)
    train_models(feature_train, feature_test, targets_train, targets_test)

    # load saved rf model
    rf_model = joblib.load(FP_MODELS + 'rfc_model.pkl')

    # compute and save results of feature importance values
    feature_importance_plot(rf_model, feature_test, FP_RESULTS)
