from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import f_regression, SelectKBest
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from dataset_management import parse_traces
warnings.filterwarnings('ignore')

IS_LOCAL = False
RANDOM_STATE = 42 if IS_LOCAL else None
EXPERIMENT = "agg_function_periodic_dataset"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/{EXPERIMENT}/{DATE}'

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
    "VOD": ['DispFrames', ]  # 'noAudioPlayed'],
}

SWITCH_PORTS = {
    "SWC1": [0, 1, 2, 3, 4],
    "SWC2": [5, 6, 7, 8, 9],
    "SWC3": [10, 11, 12, 13, 14],
    "SWC4": [15, 16, 17, 18, 19],
    "SWA1": [20, 21],
    "SWA2": [22, 23],
    "SWA3": [24, 25, 26],
    "SWA4": [27, 28],
    "SWA5": [29, 30],
    "SWA6": [31, 32, 33],
    "SWB1": [34, 35, 36],
    "SWB2": [37, 38],
    "SWB3": [39, 40, 41],
    "SWB4": [42, 43]
}

switch_from_port = {f'{port}': switch for switch,
                    ports in SWITCH_PORTS.items() for port in ports}

PER_FILES = ['X_flow.csv', 'X_port.csv']


def filter_agg_periodic(x, y, period: int, func):
    x_filtered = pd.DataFrame(columns=x.columns)
    y_filtered = pd.DataFrame(columns=y.columns)

    index = 0

    for i in range(0, len(x) - period + 1, period):
        x_filtered.loc[index] = x.iloc[i:i + period].apply(func)
        y_filtered.loc[index] = y.iloc[i:i + period].apply(func)
        index += 1

    return x_filtered, y_filtered


PERIODS = [2, 4, 8, 16, 32, 64, 128, 256]
PERIODS.reverse()

functions = [
    ('média', np.mean),
    ('máximo', np.max),
    ('mínimo', np.min)
]

for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            # smallest first
            # per switch

            x_trace, y_dataset = parse_traces(trace, y_metric, ['X_port.csv'])

            per_switch_traces = {switch: pd.DataFrame()
                                 for switch in SWITCH_PORTS.keys()}

            for feature in x_trace.columns:
                port = feature.split('_')[0]
                switch = switch_from_port[port]
                per_switch_traces[switch] = x_trace[[feature]].copy()

            for switch in per_switch_traces.keys():
                for period in PERIODS:
                    for name, func in functions:
                        x_filtered, y_filtered = filter_agg_periodic(
                            per_switch_traces[switch], y_dataset, period, func)

                        x_filtered.to_csv(
                            f'{BASE_RESULTS_PATH}/X_P-{period}_{name}_per-switch-{switch}.csv')
                        y_filtered.to_csv(
                            f'{BASE_RESULTS_PATH}/Y_P-{period}_{name}_per-switch-{switch}.csv')
                        print(
                            f'done for P-{period}_{name}_per-switch-{switch}')

            # per flow e per port
            for x_file in PER_FILES:
                x_trace_per_dataset, y_dataset = parse_traces(
                    trace, y_metric, [x_file])

                for period in PERIODS:
                    for name, func in functions:
                        x_filtered, y_filtered = filter_agg_periodic(
                            x_trace_per_dataset, y_dataset, period, func)

                        x_filtered.to_csv(
                            f'{BASE_RESULTS_PATH}/X_P-{period}_{name}_per-flow-port.csv')
                        y_filtered.to_csv(
                            f'{BASE_RESULTS_PATH}/Y_P-{period}_{name}_per-flow-port.csv')

                        print(f'done for P-{period}_{name}_per-flow-port')

            # total

            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            for period in PERIODS:
                for name, func in functions:
                    x_filtered, y_filtered = filter_agg_periodic(
                        x_trace, y_dataset, period, func)

                    x_filtered.to_csv(
                        f'{BASE_RESULTS_PATH}/X_P-{period}_{name}_total.csv')
                    y_filtered.to_csv(
                        f'{BASE_RESULTS_PATH}/Y_P-{period}_{name}_total.csv')

                    print(f'done for P-{period}_{name}_total')
