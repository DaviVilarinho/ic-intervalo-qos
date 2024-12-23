import numpy as np
import pandas as pd
from numba import njit, jit
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime
import warnings
from dataset_management import parse_traces

warnings.filterwarnings('ignore')

IS_LOCAL = False
RANDOM_STATE = 42
EXPERIMENT = "distrib_experiment_y_original_tptt"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/{EXPERIMENT}/{DATE}'

DATASET_PATH = "agg_function_periodic_dataset"

traces = {
    "VOD": [
        "VoD-BothApps-FlashcrowdLoad",
        "VoD-BothApps-PeriodicLoad",
        "VoD-SingleApp-FlashcrowdLoad",
        "VoD-SingleApp-PeriodicLoad"],
}

NROWS = None if IS_LOCAL else None

TEST_SIZE = 0.3
RANDOM_FOREST_TREES = 120


def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()


results_path = f'{BASE_RESULTS_PATH}'
for paths in [results_path]:
    try:
        os.makedirs(paths)
    except FileExistsError:
        pass

y_metrics = {
    "VOD": ['DispFrames',]  # 'noAudioPlayed'],
}


def get_per_file_name(per_file_csv):
    return f'{BASE_RESULTS_PATH}/per_{per_file_csv.split(".")[0]}.csv'


PER_FILES = ['X_flow.csv', 'X_port.csv']

for x_file in PER_FILES:
    per_dataset_file = get_per_file_name(x_file)
    with open(per_dataset_file, 'w') as f:
        f.write(f'período,função,carga,apps,feature,método,nmae,\n')

TOTAL_X_FILE_PATH = f'{BASE_RESULTS_PATH}/total_X.csv'
with open(TOTAL_X_FILE_PATH, 'w') as f:
    f.write(f'período,função,carga,apps,feature,método,nmae,\n')


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
    return skew(x)


@njit(nogil=True)
def kurtosis_numba(x):
    return kurtosis(x)


@njit(nogil=True)
def percentile_25_numba(x):
    return np.percentile(x, 25)


@njit(nogil=True)
def percentile_75_numba(x):
    return np.percentile(x, 75)


@njit(nogil=True)
def range_numba(x):
    return np.ptp(x)


def filter_agg_periodic(x, y, period: int):
    x_filtered = pd.DataFrame(columns=x.columns)
    y_filtered = pd.DataFrame(columns=y.columns)

    index = 0

    for i in range(0, len(x) - period + 1, period):
        x_filtered.loc[index] = x.iloc[i:i + period].apply(mean_numba)
        y_filtered.loc[index] = y.iloc[i:i + period].apply(mean_numba)
        index += 1

    return x_filtered, y_filtered


PERIODS = [2, 4, 8, 16, 32, 64, 128, 256]
PERIODS.reverse()

functions = [
    ('mean', mean_numba),
    ('std', std_numba),
    ('median', median_numba),
    ('skew', skew_numba),
    ('kurtosis', kurtosis_numba),
    ('25th_percentile', percentile_25_numba),
    ('75th_percentile', percentile_75_numba),
    ('range', range_numba)
]

for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            for period in PERIODS:
                for name, func in functions:
                    _, x_test, _, y_test = train_test_split(
                        x_trace, y_dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                    x_filtered, y_filtered = filter_agg_periodic(
                        x_trace, y_dataset, period)

                    x_train, _, y_train, _ = train_test_split(
                        x_filtered, y_filtered, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                    indexes_to_exclude = x_test.index

                    x_train.drop(index=indexes_to_exclude,
                                 errors='ignore', inplace=True)
                    y_train.drop(index=indexes_to_exclude,
                                 errors='ignore', inplace=True)

                    regression_tree_regressor = DecisionTreeRegressor()
                    regression_tree_regressor.fit(x_train, y_train)

                    random_forest_regressor = RandomForestRegressor(
                        n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                    random_forest_regressor.fit(x_train, y_train)

                    with open(TOTAL_X_FILE_PATH, 'a') as f:
                        f.write(
                            f'{period},{name},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                        f.write(
                            f'{period},{name},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')
                    print(f'total X {period},{name},{
                          trace_load},{trace_apps},{y_metric},RF')
