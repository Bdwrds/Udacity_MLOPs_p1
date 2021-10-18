"""
Churn prediction DS process - Functions & Solution
author: Ben E
date: 2021-10-18
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
import constants as cons

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

def perform_eda(df_eda, constants):
    '''
    perform eda on df and save figures to images folder
    input:
            df_eda: pandas dataframe
            constants: list of parameters defined in constants.py

    output:
            None
    '''
    for plot_detail in constants.dict_vars_plot.items():
        plt.figure()
        plot_var = plot_detail[0]
        plot_type = plot_detail[1]
        if plot_type == 'hist':
            df_eda[plot_var].hist()
            plt.savefig(constants.FP_EDA + 'eda_' + plot_var + '_hist.png')
        elif plot_type == 'value_counts':
            df_eda[plot_var].value_counts('normalize').plot(kind='bar')
            plt.savefig(constants.FP_EDA + 'eda_' + plot_var + '_perc.png')
        elif plot_type == 'distplot':
            sns.distplot(df_eda[plot_var])
            plt.savefig(constants.FP_EDA + 'eda_' + plot_var + '_dist.png')
        else:
            print('ERROR: plot type not listed')
        plt.close()

    plt.figure()
    sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(constants.FP_EDA + 'eda_all_heatmap.png')

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
              keep_cols: the columns kept from df_eng
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
                                constants):
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
            constants: list of parameters defined in constants.py

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
    plt.savefig(constants.PLOT_MODEL_1_RESULTS)
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
    plt.savefig(constants.PLOT_MODEL_2_RESULTS)
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
    plt.savefig(output_pth)
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
    plt.savefig(output_pth)
    print("saved feature_importance_values")


def feature_importance_plot(model, x_data, constants):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    shap_importance_plot(model, x_data, constants.PLOT_SHAP)
    model_feat_importance_plot(model, x_data, constants.PLOT_FEAT_IMP)


def train_models(x_train, x_test, y_train, y_test, constants):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
              constants: list of parameters defined in constants.py
    output:
              None
    '''
    print("\nTraining models...")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    print('\nTraining RF..')
    cv_rfc = GridSearchCV(estimator=rfc, param_grid = constants.param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    print('\nTraining LR..')
    lrc.fit(x_train, y_train)

    # plots
    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    ax_gca = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=ax_gca,
        alpha=0.8)
    lrc_plot.plot(ax=ax_gca, alpha=0.8)
    plt.savefig(constants.PLOT_ROC)
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, constants.MODEL_NAME_1)
    joblib.dump(lrc, constants.MODEL_NAME_2)
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
        constants)


# run if not imported
if __name__ == "__main__":
    # load data
    df = import_data(pth = cons.FILE_NAME)

    # create target from adjustment variables
    df[cons.TARGET_COL] = df[cons.VAR_1].apply(
        lambda val: 0 if val == cons.FIELD_1 else 1)

    # perform eda
    perform_eda(df, cons)

    # encode categorical variables
    df_encoded = encoder_helper(df, cons.lst_features_encode, cons.TARGET_COL)

    # perform feat engineering
    feature_train, feature_test, targets_train, targets_test = perform_feature_engineering(
        df_encoded, cons.lst_keep_features, cons.TARGET_COL)

    # train models
    train_models(feature_train, feature_test, targets_train, targets_test, cons)

    # load saved rf model
    rf_model = joblib.load(cons.MODEL_NAME_1)

    # compute and save results of feature importance values
    feature_importance_plot(rf_model, feature_test, cons)
