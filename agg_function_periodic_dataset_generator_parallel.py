import pandas as pd
import os
import numpy as np
import warnings

from dataset_management import parse_traces
warnings.filterwarnings('ignore')

IS_LOCAL = False
RANDOM_STATE = 42 if IS_LOCAL else None
EXPERIMENT = "agg_function_periodic_dataset"

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

from numba import njit, jit, prange

def filter_agg_periodic(x, y, period: int, func):
    x_agg = x.rolling(period, step=period).apply(func, engine='numba', raw=True).dropna()
    y_agg = y.rolling(period, step=period).apply(func, engine='numba', raw=True).dropna()
    
    return x_agg, y_agg

PERIODS = [2, 4, 8, 16, 32, 64, 128, 256]
PERIODS.reverse()

@njit(nogil=True)
def mean_numba(x):
    return np.sum(x) / x.size

@jit(parallel=True)
def max_numba(a):
    return np.max(a)

@jit(parallel=True)
def min_numba(a):
    return np.min(a)

functions = [
    ('média', mean_numba),
    ('máximo', max_numba),
    ('mínimo', min_numba)
]

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
                for name, func in functions:
                    x_filtered, y_filtered = filter_agg_periodic(
                        x_trace, y_dataset, period, func)

                    x_filtered.to_csv(f'{BASE_RESULTS_PATH}/X_{trace}_P-{period}_{name}_total.csv')
                    y_filtered.to_csv(f'{BASE_RESULTS_PATH}/Y_{trace}_P-{period}_{name}_total.csv')
            
                    print(f'done for P-{trace}_{period}_{name}_total')
