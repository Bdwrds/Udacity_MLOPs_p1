"""
Script to log and test churn_library.oy
author: Ben E
date: 2021-10-14
"""
import os
import logging
import numpy as np
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "Testing import_data: SUCCESS: dataframe shape %s" %
            (str(
                df.shape)))
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns: %s"
                      (str(df.shape)))
        raise err

    return df


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = test_import(cls.import_data)
    eda_fp = './images/eda/'

    try:
        assert os.path.isdir(eda_fp) is not False
        logging.info('Testing perform_eda: SUCCESS - Filepath exists')
    except AssertionError as err:
        logging.error("Testing perform_eda: FAILURE - Filepath doesnt exist")
        raise err

    try:
        assert 'Attrition_Flag' in df.columns
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info('Testing perform_eda: SUCCESS - Target exists')
    except AssertionError as err:
        logging.error("Testing perform_eda: FAILURE - Target doesnt exist")
        raise err

    plot_list = [
        'Total_Trans_Ct',
        'Marital_Status',
        'Customer_Age',
        'Churn'
    ]
    try:
        assert set(plot_list).issubset(df.columns) is not False
        logging.info('Testing perform_eda: SUCCESS - All features exist')
    except AssertionError as err:
        logging.info(
            'Testing perform_eda: FAILURE - Features(s) for plots dont exist')
        raise err

    try:
        perform_eda(df)
        logging.info('Testing perform_eda: SUCCESS - Completed')
    except KeyError as err:
        logging.info('Testing perform_eda: FAILURE - Failed to complete')
        raise err
    return df


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df_encode = test_eda(cls.perform_eda)

    try:
        assert 'Attrition_Flag' in df_encode.columns
        df_encode['Churn'] = df_encode['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info('Testing test_encoder_helper: SUCCESS - Target exists')
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: FAILURE - Target doesnt exist")
        raise err

    target_col = 'Churn'
    variable_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    try:
        assert set(variable_lst).issubset(df_encode.columns)
        assert target_col in df_encode.columns
        logging.info(
            'Testing encoder_helper: SUCCESS - All features + target exist in df')
    except AssertionError as err:
        logging.info(
            'Testing encoder_helper: FAILURE - Features(s) or target dont exist in df')
        raise err

    # which features are categories
    feature_bool_ls = [df_encode[feat].dtype == np.object for feat in variable_lst]

    try:
        # check they are all categories
        assert all(feature_bool_ls) is True
        logging.info(
            'Testing encoder_helper: SUCCESS - All features are categories')
    except AssertionError as err:
        logging.info(
            'Testing encoder_helper: FAILURE - Some feature(s) arent categories')
        raise err

    try:
        df_encoded_rtn = encoder_helper(df_encode, variable_lst, target_col)
        logging.info('Testing encoder_helper: SUCCESS - Completed')
    except KeyError as err:
        logging.info('Testing encoder_helper: FAILURE - Failed to complete')
        raise err

    return df_encoded_rtn


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df_encoded = test_encoder_helper(cls.encoder_helper)
        logging.info(
            'Testing test_perform_feature_engineering: SUCCESS - df_encoded loads empty')
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
    except AssertionError as err:
        logging.info(
            'Testing test_perform_feature_engineering: FAILURE - df_encoded doesnt load')
        raise err

    target_col = 'Churn'
    feature_lst = [
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
        'Card_Category_Churn'
    ]

    try:
        assert set(feature_lst).issubset(df_encoded.columns)
        assert target_col in df_encoded.columns
        logging.info(
            'Testing test_perform_feature_engineering: SUCCESS \
            - All features + target exist in df_encoded'
        )
    except AssertionError as err:
        logging.info(
            'Testing test_perform_feature_engineering: FAILURE \
            - Features(s) or target dont exist in df_encoded'
        )
        raise err

    feature_train, feature_test, targets_train, targets_test = perform_feature_engineering(
        df_encoded, feature_lst, target_col)
    return feature_train, feature_test, targets_train, targets_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        feature_train_rtn, feature_test_rtn, targets_train_rtn, targets_test_rtn = \
        test_perform_feature_engineering(cls.perform_feature_engineering)
        assert feature_train_rtn.shape[0] == targets_train_rtn.shape[0]
        assert feature_test_rtn.shape[0] == targets_test_rtn.shape[0]
        assert feature_train_rtn.shape[0] > 0
        assert feature_test_rtn.shape[0] > 0
        logging.info(
            'Testing test_train_models: SUCCESS - train and test sets load'
        )
    except AssertionError as err:
        logging.info(
            'Testing test_train_models: FAILURE - check shape of train/ test sets '
        )
        raise err

    train_models(feature_train_rtn, feature_test_rtn, targets_train_rtn, targets_test_rtn)


if __name__ == "__main__":
    test_train_models(cls.train_models)
