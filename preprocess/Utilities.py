from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import inspect


def split_train_test(data, test_proportion):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if test_proportion <= 0 or test_proportion >= 1:
        raise ValueError(form_error_msg("Invalid parameter test_proportion."))
    return train_test_split(data, test_size=test_proportion)


def get_function_name():
    currentframe = inspect.currentframe()
    return inspect.getframeinfo(currentframe).function


def form_error_msg(error_msg):
    return get_function_name() + ":" + error_msg


def get_data_frame_row_count(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    return data_frame.shape[0]


def get_data_frame_col_count(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if len(data_frame) == 1:
        return 1
    return data_frame.shape[1]


def get_data_frame_col_names(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    return data_frame.columns


def filter_col_data(data, cols_array):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if is_valid_list(cols_array) is False:
        raise ValueError(form_error_msg("Invalid parameter cols_array."))
    return data.loc[:, cols_array]


def is_data_frame(data):
    return isinstance(data, pd.DataFrame)


def is_valid_list(array):
    return isinstance(array, list) and array


def one_hot_encode_cols(data, cols):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if is_valid_list(cols) is False:
        raise ValueError(form_error_msg("Invalid parameter cols."))
    transformer = make_column_transformer((OneHotEncoder(), cols), remainder="passthrough")
    return transformer.fit_transform(data)



