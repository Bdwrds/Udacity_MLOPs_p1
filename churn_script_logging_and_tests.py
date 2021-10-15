"""
Script to log and test churn_library.oy
author: Ben E
date: 2021-10-14
"""
import os
import logging
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
        logging.info("Testing dataframe shape: SUCCESS: %s" % (str(df.shape)))
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns: %s"
                      (str(df.shape)))
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    eda_fp = './images/eda/'

    try:
        assert os.path.isdir(eda_fp) is not False
        logging.info('Testing perform_eda: SUCCESS - Filepath exists')
    except AssertionError as err:
        logging.error("Testing perform_eda: FAILURE - Filepath doesnt exist")
        raise err

    df = cls.import_data("./data/bank_data.csv")

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
        encoded_df = perform_eda(df)
        logging.info('Testing perform_eda: SUCCESS - Completed')
    except KeyError as err:
        logging.info('Testing perform_eda: FAILURE - Failed to complete')
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    import numpy as np
    df = cls.import_data(pth=r"./data/bank_data.csv")

    try:
        assert 'Attrition_Flag' in df.columns
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info('Testing test_encoder_helper: SUCCESS - Target exists')
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: FAILURE - Target doesnt exist")
        raise err

    target_col = 'Churn'
    feature_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    try:
        assert set(feature_lst).issubset(df.columns)
        assert target_col in df.columns
        logging.info(
            'Testing encoder_helper: SUCCESS - All features + target exist in df')
    except AssertionError as err:
        logging.info(
            'Testing encoder_helper: FAILURE - Features(s) or target dont exist in df')
        raise err

    # which features are categories
    feature_bool_ls = [df[feat].dtype == np.object for feat in feature_lst]

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
        encoder_helper(df, feature_lst, target_col)
        logging.info('Testing encoder_helper: SUCCESS - Completed')
    except KeyError as err:
        logging.info('Testing encoder_helper: FAILURE - Failed to complete')
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data(pth=r"./data/bank_data.csv")
    target_col = 'Churn'
    feature_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = cls.encoder_helper(df, feature_lst, target_col)
    perform_feature_engineering(df_encoded, feature_lst, target_col)


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    # test_perform_feature_engineering(cls.perform_feature_engineering)
