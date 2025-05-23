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

IS_LOCAL = False # os.uname()[1].split('-').pop(0) == "ST"
RANDOM_STATE = 42
EXPERIMENT = "agg_function_periodic_experiment_y_original_tptt"

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
    "VOD": ['DispFrames',]# 'noAudioPlayed'],
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


MINIMAL_PATH = f'{BASE_RESULTS_PATH}/minimal_with_univariate.csv'
with open(MINIMAL_PATH, 'w') as f:
    f.write(f'período,função,carga,apps,feature,método,nmae\n')

BEST_K_PATH = f'{BASE_RESULTS_PATH}/best_k.csv'
with open(BEST_K_PATH, 'w') as f:
    f.write(f'período,função,Features,\n')


def filter_agg_periodic(x, y, period: int, func):
    x_filtered = pd.DataFrame(columns=x.columns)
    y_filtered = pd.DataFrame(columns=y.columns)

    index = 0

    for i in range(0, len(x) - period + 1, period):
        x_filtered.loc[index] = x.iloc[i:i + period].apply(func)
        y_filtered.loc[index] = y.iloc[i:i + period].apply(func)
        index += 1

    return x_filtered, y_filtered

PERIODS = [2,4,8,16,32,64,128,256]
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
            # total

            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            for period in PERIODS:
                for name, func in functions:
                    _, x_test, _, y_test = train_test_split(
                        x_trace, y_dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                    x_filtered = pd.read_csv(
                        f'{DATASET_PATH}/X_{trace}_P-{period}_{name}_total.csv', index_col="TimeStamp")
                    y_filtered = y_dataset.loc[x_filtered.index]

                    x_train, _, y_train, _ = train_test_split(
                        x_filtered, y_filtered, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                    x_test = x_test.iloc[period:].sample(n=int(len(x_train) * 0.3 / 0.7), random_state=RANDOM_STATE)
                    y_test = y_test.loc[x_test.index]
                    original_indices = x_trace.index.get_indexer(x_test.index)
                    x_test = x_filtered.iloc[((original_indices // period) - 1).clip(0)]

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
                    print(f'total X {period},{name},{trace_load},{trace_apps},{y_metric},RF')


                    # minimal
                    k = 12
                    best_k = []
                    selectK = SelectKBest(f_regression, k=k)

                    selectK.set_output(transform="pandas")

                    minimal_dataset = selectK.fit_transform(x_train, y_train)
                    best_k.append(list(minimal_dataset.columns))

                    x_test_minimal = x_test[minimal_dataset.columns]

                    regression_tree_regressor = DecisionTreeRegressor()
                    regression_tree_regressor.fit(minimal_dataset, y_train)

                    random_forest_regressor = RandomForestRegressor(
                        n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                    random_forest_regressor.fit(minimal_dataset, y_train)

                    with open(MINIMAL_PATH, 'a') as f:
                        f.write(
                            f'{period},{name},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test_minimal), y_test[y_metric])},\n')
                        f.write(
                            f'{period},{name},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test_minimal), y_test[y_metric])},\n')

                    print(f'univariate {period},{name},{trace_load},{trace_apps},{y_metric},RF')

                    with open(BEST_K_PATH, 'a') as f:
                        f.write(f'{period},{name},{best_k},\n')