import numpy as np

def mean_absolute_error(y_true, y_pred):        # y values have the yield
    true = np.array(y_true)
    pred = np.array(y_pred)

    # MAE formula: 1/n * (summation(true_y - pred_y))
    # use numpy to subtract each item from their respective counterparts in array

    absolute_errors = np.abs(true - pred)

    return np.mean(absolute_errors)