from numba import njit, jit, prange
import pandas as pd
import os
import numpy as np
import warnings
from scipy.stats import skew, kurtosis

from dataset_management import parse_traces
warnings.filterwarnings('ignore')

IS_LOCAL = False
RANDOM_STATE = 42
EXPERIMENT = "distrib_function_periodic_dataset"

PASQUINIS_PATH = "../traces-netsoft-2017"
BASE_RESULTS_PATH = f'{EXPERIMENT}'

traces = {
    "VOD": [
        "VoD-BothApps-FlashcrowdLoad",
        "VoD-BothApps-PeriodicLoad",
        "VoD-SingleApp-FlashcrowdLoad",
        "VoD-SingleApp-PeriodicLoad"],
}

results_path = f'{BASE_RESULTS_PATH}'
for paths in [results_path]:
    try:
        os.makedirs(paths)
    except FileExistsError:
        pass

y_metrics = {
    "VOD": ['DispFrames', ]  # 'noAudioPlayed'],
}


def filter_agg_periodic(x, y, period: int):
    x_agg = x.rolling(period, step=period).apply(
        mean_numba, engine='numba', raw=True).dropna()
    y_agg = y.rolling(period, step=period).apply(
        mean_numba, engine='numba', raw=True).dropna()

    x_std = x.rolling(period, step=period).apply(
        std_numba, engine='numba', raw=True).dropna()
    y_std = y.rolling(period, step=period).apply(
        std_numba, engine='numba', raw=True).dropna()

    x_median = x.rolling(period, step=period).apply(
        median_numba, engine='numba', raw=True).dropna()
    y_median = y.rolling(period, step=period).apply(
        median_numba, engine='numba', raw=True).dropna()

    x_skew = x.rolling(period, step=period).apply(
        skew_numba, engine='numba', raw=True).dropna()
    y_skew = y.rolling(period, step=period).apply(
        skew_numba, engine='numba', raw=True).dropna()

    x_kurt = x.rolling(period, step=period).apply(
        kurtosis_numba, engine='numba', raw=True).dropna()
    y_kurt = y.rolling(period, step=period).apply(
        kurtosis_numba, engine='numba', raw=True).dropna()

    x_25th = x.rolling(period, step=period).apply(
        percentile_25_numba, engine='numba', raw=True).dropna()
    y_25th = y.rolling(period, step=period).apply(
        percentile_25_numba, engine='numba', raw=True).dropna()

    x_75th = x.rolling(period, step=period).apply(
        percentile_75_numba, engine='numba', raw=True).dropna()
    y_75th = y.rolling(period, step=period).apply(
        percentile_75_numba, engine='numba', raw=True).dropna()

    x_range = x.rolling(period, step=period).apply(
        range_numba, engine='numba', raw=True).dropna()
    y_range = y.rolling(period, step=period).apply(
        range_numba, engine='numba', raw=True).dropna()

    x_dataset = pd.concat([x_agg, x_std, x_median, x_skew, x_kurt, x_25th, x_75th, x_range], axis=1)
    y_dataset = pd.concat([y_agg, y_std, y_median, y_skew, y_kurt, y_25th, y_75th, y_range], axis=1)

    return x_dataset, y_dataset


PERIODS = [
    # 2,
    # 4,
    8,
    16, 32, 64, 128, 256]
PERIODS.reverse()


@njit(nogil=True)
def mean_numba(x):
    return np.sum(x) / x.size


@njit(nogil=True)
def std_numba(x):
    return np.sqrt(np.sum((x - mean_numba(x)) ** 2) / x.size)


@njit(nogil=True)
def median_numba(x):
    return np.median(x)

@njit(nogil=True)
def skew_numba(x):
    mean_x = mean_numba(x)
    std_x = std_numba(x)
    return np.sum(((x - mean_x) / std_x) ** 3) / x.size


@njit(nogil=True)
def kurtosis_numba(x):
    mean_x = mean_numba(x)
    std_x = std_numba(x)
    return np.sum(((x - mean_x) / std_x) ** 4) / x.size - 3


@njit(nogil=True)
def percentile_25_numba(x):
    return np.percentile(x, 25)


@njit(nogil=True)
def percentile_75_numba(x):
    return np.percentile(x, 75)


@njit(nogil=True)
def range_numba(x):
    return np.ptp(x)


for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            # total

            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            for period in PERIODS:
                x_filtered, y_filtered = filter_agg_periodic(
                    x_trace, y_dataset, period)

                x_filtered.to_csv(
                    f'{BASE_RESULTS_PATH}/X_{trace}_P-{period}_total.csv')
                y_filtered.to_csv(
                    f'{BASE_RESULTS_PATH}/Y_{trace}_P-{period}_total.csv')

                print(f'done for P-{trace}_{period}_total')
