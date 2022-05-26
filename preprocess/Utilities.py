from sklearn.model_selection import train_test_split
import inspect


def split_train_test(data, test_proportion):
    if (data == None):
        raise ValueError(form_error_msg("Invalid parameter data."))
    if (test_proportion == None or test_proportion <= 0 or test_proportion >= 1):
        raise ValueError(form_error_msg("Invalid parameter test_proportion."))
    return train_test_split(data, test_size=test_proportion)


def get_function_name():
    currentframe = inspect.currentframe()
    return inspect.getframeinfo(currentframe).function


def form_error_msg(error_msg):
    return get_function_name() + ":" + error_msg

def get_data_frame_row_count(data_frame):
    if (data_frame == None):
        raise ValueError(form_error_msg("Invalid parameter data."))
    return data_frame.shape[0]

def get_data_frame_col_count(data_frame):
    if (data_frame == None):
        raise ValueError(form_error_msg("Invalid parameter data."))
    if (len(data_frame) == 1):
        return 1
    return data_frame.shape[1]

def get_data_frame_col_names(data_frame):
    if (data_frame == None):
        raise ValueError(form_error_msg("Invalid parameter data."))
    return data_frame.columns

# def filter_col_data(data,columns):
#     if da:



