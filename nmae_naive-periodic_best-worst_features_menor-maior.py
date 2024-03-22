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


from dataset_management import parse_traces
warnings.filterwarnings('ignore')

IS_LOCAL = False
RANDOM_STATE = 42 if IS_LOCAL else None
EXPERIMENT = "nmae_naive-periodic_best-worst_features_menor-maior"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'/home/dv/data/projects/ic-experiments/{EXPERIMENT}/{DATE}'

traces = {
    "VOD": [
        "VoD-BothApps-FlashcrowdLoad",
        "VoD-BothApps-PeriodicLoad",
        "VoD-SingleApp-FlashcrowdLoad",
        "VoD-SingleApp-PeriodicLoad"],
    "KV": [
        "KV-BothApps-FlashcrowdLoad",
        "KV-BothApps-PeriodicLoad",
        "KV-SingleApp-FlashcrowdLoad",
        "KV-SingleApp-PeriodicLoad"]
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
    "VOD": ['DispFrames', 'noAudioPlayed'],
    "KV": ["ReadsAvg", "WritesAvg"]
}

TOTAL_X_FILE_PATH = f'{BASE_RESULTS_PATH}/total_X.csv'
with open(TOTAL_X_FILE_PATH, 'w') as f:
    f.write(f'carga,apps,y,método,período,tipo x,período que ocorreu,x,condição,nmae,\n')


def filter_periodic(x, y, period: int):
    x = x[x.index % period == 0]
    y = y[y.index % period == 0]

    return x, y


PERIODS = [1,2, 4, 8, 16, 32, 64, 128, 256]


for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            worst_100_in = 1
            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])
            for worst_metric in ['5_eth1_rxkB.s',
                                 '2_kbmemfree',
                                 '3_cpu21_.idle',
                                 '3_cpu21_.idle',
                                 '3_cpu14_.usr',
                                 '3_cpu11_.usr',
                                 '3_cpu18_.usr',
                                 '3_all_..usr',
                                 '3_cpu16_.idle']:

                for period in PERIODS:
                    x_filtered, y_filtered = filter_periodic(
                        x_trace, y_dataset, period)

                    xmenorque15 = x_filtered[y_filtered[y_metric] < 15]
                    ymenorque15 = y_filtered[y_filtered[y_metric] < 15]

                    x_feature = pd.DataFrame(xmenorque15[worst_metric])

                    try:
                        x_train, x_test, y_train, y_test = train_test_split(
                            x_feature, ymenorque15, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                        regression_tree_regressor = DecisionTreeRegressor()
                        regression_tree_regressor.fit(x_train, y_train)

                        random_forest_regressor = RandomForestRegressor(
                            n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                        random_forest_regressor.fit(x_train, y_train)

                        with open(TOTAL_X_FILE_PATH, 'a') as f:
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RT,{period},worst_100,{worst_100_in},{worst_metric},y < 15,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RF,{period},worst_100,{worst_100_in},{worst_metric},y < 15,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')
                    except ValueError:
                        pass

                    try:
                        xmaiorque20 = x_filtered[y_filtered[y_metric] > 20]
                        ymaiorque20 = y_filtered[y_filtered[y_metric] > 20]

                        x_feature = pd.DataFrame(xmaiorque20[worst_metric])

                        x_train, x_test, y_train, y_test = train_test_split(
                            x_feature, ymaiorque20, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                        regression_tree_regressor = DecisionTreeRegressor()
                        regression_tree_regressor.fit(x_train, y_train)

                        random_forest_regressor = RandomForestRegressor(
                            n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                        random_forest_regressor.fit(x_train, y_train)

                        with open(TOTAL_X_FILE_PATH, 'a') as f:
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RT,{period},worst_100,{worst_100_in},{worst_metric},y > 20,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RF,{period},worst_100,{worst_100_in},{worst_metric},y > 20,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')
                    except ValueError:
                        pass
                    print(trace, period, worst_metric)
                worst_100_in *= 2

            best_100_in = 1
            for best_metric in [
                "3_cpu21_.idle",
                "4_cpu13_.idle",
                "4_ldavg.5",
                "3_ldavg.1",
                "3_ldavg.1",
                "4_ldavg.1",
                "4_pgfree.s",
                "4_pgfree.s",
                "3_i128_intr.s"]:

                for period in PERIODS:
                    x_filtered, y_filtered = filter_periodic(
                        x_trace, y_dataset, period)

                    xmenorque15 = x_filtered[y_filtered[y_metric] < 15]
                    ymenorque15 = y_filtered[y_filtered[y_metric] < 15]

                    x_feature = pd.DataFrame(xmenorque15[best_metric])

                    try:
                        x_train, x_test, y_train, y_test = train_test_split(
                            x_feature, ymenorque15, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                        regression_tree_regressor = DecisionTreeRegressor()
                        regression_tree_regressor.fit(x_train, y_train)

                        random_forest_regressor = RandomForestRegressor(
                            n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                        random_forest_regressor.fit(x_train, y_train)

                        with open(TOTAL_X_FILE_PATH, 'a') as f:
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RT,{period},best,{best_100_in},{best_metric},y < 15,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RF,{period},best,{best_100_in},{best_metric},y < 15,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')
                    except ValueError:
                        pass

                    try:
                        xmaiorque20 = x_filtered[y_filtered[y_metric] > 20]
                        ymaiorque20 = y_filtered[y_filtered[y_metric] > 20]

                        x_feature = pd.DataFrame(xmaiorque20[best_metric])

                        x_train, x_test, y_train, y_test = train_test_split(
                            x_feature, ymaiorque20, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                        regression_tree_regressor = DecisionTreeRegressor()
                        regression_tree_regressor.fit(x_train, y_train)

                        random_forest_regressor = RandomForestRegressor(
                            n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                        random_forest_regressor.fit(x_train, y_train)

                        with open(TOTAL_X_FILE_PATH, 'a') as f:
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RT,{period},worst_100,{worst_100_in},{worst_metric},y > 20,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                            f.write(
                                f'{trace_load},{trace_apps},{y_metric},RF,{period},worst_100,{worst_100_in},{worst_metric},y > 20,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')
                        print(trace, period, best_metric)
                    except ValueError:
                        pass
                best_100_in *= 2

            del x_trace, y_dataset
