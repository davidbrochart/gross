from pandas import DataFrame
import pandas as pd
import math
import numpy as np

def mse(x_obs, x_est, func = None):
    df = DataFrame({'x_obs': x_obs, 'x_est': x_est})
    if func != None:
        df = df.apply(func)
    df = df.dropna()
    return np.mean(np.square(df.x_obs.values - df.x_est.values))

def rmse(x_obs, x_est, func = None):
    df = DataFrame({'x_obs': x_obs, 'x_est': x_est})
    if func != None:
        df = df.apply(func)
    df = df.dropna()
    return math.sqrt(np.mean(np.square(df.x_obs.values - df.x_est.values)))

def nse(x_obs, x_est, func = None, warmup = 0):
    df = DataFrame({'x_obs': x_obs[warmup:], 'x_est': x_est[warmup:]})
    if func != None:
        df = df.apply(func)
    df = df.dropna()
    return 1. - (np.sum(np.square(df.x_obs.values - df.x_est.values)) / np.sum(np.square(df.x_obs.values - np.mean(df.x_obs.values))))

def nse_min(x_obs, x_est, func = None, warmup = 0):
    return 1. - nse(x_obs, x_est, func, warmup)

def calibration(x, in_obs, out_obs, warmup_period, crit_func, model, x_range, func = None, init_conf = None, init_state = None):
    for i in range(len(x_range)): # forbidden x values
        if x_range[i][0] != None:
            if x[i] < x_range[i][0]:
                return np.inf
        if x_range[i][1] != None:
            if x[i] > x_range[i][1]:
                return np.inf
    data_nb = out_obs.size
    q_mod = model(x)
    if init_conf != None:
        q_mod.set_conf(init_conf)
    if init_state != None:
        q_mod.set_state(init_state)
    out_sim = q_mod.run(in_obs)
    error = crit_func(out_obs[warmup_period:], out_sim[warmup_period:], func)
    return error
