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
EXPERIMENT = "distrib_experiment_y_original"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/{EXPERIMENT}/{DATE}'

DATASET_PATH = "distrib_function_periodic_dataset"

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



PER_FILES = ['X_flow.csv', 'X_port.csv']

TOTAL_X_FILE_PATH = f'{BASE_RESULTS_PATH}/total_X.csv'
with open(TOTAL_X_FILE_PATH, 'w') as f:
    f.write(f'período,carga,apps,feature,método,nmae,\n')


PERIODS = [8, 16, 32, 64, 128, 256]
PERIODS.reverse()

for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            for period in PERIODS:
                x_filtered = pd.read_csv(
                    f'{DATASET_PATH}/X_{trace}_P-{period}_total.csv', index_col="TimeStamp")
                y_filtered = y_dataset.loc[x_filtered.index]

                x_train, x_test, y_train, y_test = train_test_split(
                    x_filtered, y_filtered, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                regression_tree_regressor = DecisionTreeRegressor()
                regression_tree_regressor.fit(x_train, y_train)

                random_forest_regressor = RandomForestRegressor(
                    n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                random_forest_regressor.fit(x_train, y_train)

                with open(TOTAL_X_FILE_PATH, 'a') as f:
                    f.write(
                        f'{period},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                    f.write(
                        f'{period},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')
                print(f'total X {period},{trace_load},{trace_apps},{y_metric}')
