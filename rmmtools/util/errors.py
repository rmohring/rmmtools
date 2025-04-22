#!/usr/bin/env python
import numpy as np
import sklearn.metrics as skm

rmsle = skm.root_mean_squared_log_error
rmse = skm.root_mean_squared_error
mse = skm.mean_squared_error
mad = skm.mean_absolute_error


def mape(*args, **kwargs):
    return skm.mean_absolute_percentage_error(*args, **kwargs) * 100


def mean_logloss(y_true, y_pred):
    return np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))


# def rmsle(y_true, y_pred):
#    #return np.sqrt(sklearn.metrics.mean_squared_log_error(y_true, y_pred))

# def rmse(y_true, y_pred):
#    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# def mad(y_true, y_pred):
#    return np.mean(np.abs(y_true - y_pred))

# def mape(y_true, y_pred):
#    y_true, y_pred = np.array(y_true), np.array(y_pred)
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
