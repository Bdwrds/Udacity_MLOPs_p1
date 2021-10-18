"""
Script to log and test churn_library.py
author: Ben E
date: 2021-10-18
"""
import os
import logging
import numpy as np
import churn_library as cls
import constants as cons

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data, constants):
    '''
    test data import
    input:
        encoder_helper: function from churn_library.py
        constants: list of parameters defined in constants.py
    output: df_load: pandas dataframe
    '''
    # check the csv file exists
    try:
        df_load = import_data(constants.FILE_NAME)
        logging.info("Testing import_data: SUCCESS - Loaded data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    # check the shape of the csv is not empty
    try:
        assert df_load.shape[0] > 0
        assert df_load.shape[1] > 0
        logging.info(
            "Testing import_data: SUCCESS: dataframe shape %s", str(df_load.shape))
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns:\
             %s", str(df_load.shape))
        raise err

    # return the df_load for next test
    return df_load

def test_eda(perform_eda, constants):
    '''
    test perform_eda function
    input:
        encoder_helper: function from churn_library.py
        constants: list of parameters defined in constants.py
    output: df_adj: pandas dataframe
    '''
    # run import test to obtain data
    try:
        df_adj = test_import(cls.import_data, constants)
        logging.info('Testing perform_eda: SUCCESS - test_import completes')
    except AttributeError as err:
        logging.info('Testing perform_eda: FAILURE - test_import fails')
        raise err

    # check the file path exists
    try:
        assert os.path.isdir(constants.FP_EDA) is not False
        logging.info('Testing perform_eda: SUCCESS - Filepath exists')
    except AssertionError as err:
        logging.error("Testing perform_eda: FAILURE - Filepath doesnt exist")
        raise err

    # create new target feature
    try:
        assert constants.VAR_1 in df_adj.columns
        df_adj.loc[:,constants.TARGET_COL] = df_adj[:, constants.VAR_1].apply(
            lambda val: 0 if val == constants.VAR_2 else 1)
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
    # check necessary features exist for ED Analysis plots
    try:
        assert set(plot_list).issubset(df_adj.columns) is not False
        logging.info('Testing perform_eda: SUCCESS - All features exist')
    except AssertionError as err:
        logging.info(
            'Testing perform_eda: FAILURE - Features(s) for plots dont exist')
        raise err

    # perform the eda
    try:
        perform_eda(df_adj, constants)
        logging.info('Testing perform_eda: SUCCESS - Completed')
    except KeyError as err:
        logging.info('Testing perform_eda: FAILURE - Failed to complete')
        raise err

    # return the df for next test
    return df_adj


def test_encoder_helper(encoder_helper, constants):
    '''
    test encoder helper
    input:
        encoder_helper: function from churn_library.py
        constants: list of parameters defined in constants.py
    output:
        df_encoded_rtn: dataframe with encoded variables
    '''
    # perform previous test to obtain checked df
    df_encode = test_eda(cls.perform_eda, constants)

    # check the encoded features are in the df
    try:
        assert set(constants.lst_features_encode).issubset(df_encode.columns)
        assert constants.TARGET_COL in df_encode.columns
        logging.info(
            'Testing encoder_helper: SUCCESS - All features + target exist in df')
    except AssertionError as err:
        logging.info(
            'Testing encoder_helper: FAILURE - Features(s) or target dont exist in df')
        raise err

    # check that these features are categories
    feature_bool_ls = \
    [df_encode.loc[:,feat].dtype == np.object for feat in constants.lst_features_encode]
    try:
        assert all(feature_bool_ls) is True
        logging.info(
            'Testing encoder_helper: SUCCESS - All features are categories')
    except AssertionError as err:
        logging.info(
            'Testing encoder_helper: FAILURE - Some feature(s) arent categories')
        raise err

    # encode those features
    try:
        df_encoded_rtn = \
        encoder_helper(df_encode, constants.lst_features_encode, constants.TARGET_COL)
        logging.info('Testing encoder_helper: SUCCESS - Completed')
    except KeyError as err:
        logging.info('Testing encoder_helper: FAILURE - Failed to complete')
        raise err

    # return encoded df for next test
    return df_encoded_rtn


def test_perform_feature_engineering(perform_feature_engineering, constants):
    '''
    test perform_feature_engineering
    input:
        perform_feature_engineering: function from churn_library.py
        constants: list of parameters defined in constants.py
    output:
        feature_train: dataframe with training features
        feature_test: dataframe with test features
        targets_train: dataframe with training target
        targets_test: dataframe with test target
    '''
    try:
        df_encoded = test_encoder_helper(cls.encoder_helper, constants)
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
        logging.info(
            'Testing test_perform_feature_engineering: SUCCESS - df_encoded loads')
    except AssertionError as err:
        logging.info(
            'Testing test_perform_feature_engineering: FAILURE - df_encoded doesnt load')
        raise err

    # check final features exist in encoded df
    try:
        assert set(constants.lst_keep_features).issubset(df_encoded.columns)
        assert constants.TARGET_COL in df_encoded.columns
        logging.info(
            'Testing test_perform_feature_engineering: SUCCESS - All vars exists in df'
        )
    except AssertionError as err:
        logging.info(
            'Testing test_perform_feature_engineering: FAILURE - Check vars exists in df'
        )
        raise err

    # return dataframes for modelling
    feature_train, feature_test, targets_train, targets_test = perform_feature_engineering(
        df_encoded, constants.lst_keep_features, constants.TARGET_COL)

    return feature_train, feature_test, targets_train, targets_test


def test_train_models(train_models, constants):
    '''
    test train_models
    '''
    fp_models = constants.FP_MODELS
    fp_results = constants.FP_RESULTS

    # obtain dataframes for modelling and validate sizes
    try:
        feature_train_rtn, feature_test_rtn, targets_train_rtn, targets_test_rtn = \
        test_perform_feature_engineering(cls.perform_feature_engineering, constants)
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

    # check the file paths for models & results exist
    try:
        assert os.path.isdir(fp_models) is not False
        assert os.path.isdir(fp_results) is not False
        logging.info('Testing test_train_models: SUCCESS - Filepath exists')
    except AssertionError as err:
        logging.info('Testing test_train_models: FAILURE - Filepaths do not exist')
        raise err

    # train models
    train_models(
        feature_train_rtn,
        feature_test_rtn,
        targets_train_rtn,
        targets_test_rtn,
        constants
    )

if __name__ == "__main__":
    test_train_models(cls.train_models, cons)
