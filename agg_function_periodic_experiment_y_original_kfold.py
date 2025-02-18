import numpy as np
from sklearn.feature_selection import f_regression, SelectKBest
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import os
import numpy as np
from datetime import datetime
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from dataset_management import parse_traces
warnings.filterwarnings('ignore')

IS_LOCAL = False  # os.uname()[1].split('-').pop(0) == "ST"
RANDOM_STATE = 42
EXPERIMENT = "agg_function_periodic_experiment_y_original_k_fold"

K_FOLD = 3

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/{EXPERIMENT}/{DATE}'

AGG_DATASET_PATH = "agg_function_periodic_dataset"

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

TOTAL_X_FILE_PATH = f'{BASE_RESULTS_PATH}/total_X.csv'
with open(TOTAL_X_FILE_PATH, 'w') as f:
    f.write(f'período,função,carga,apps,feature,método,nmae,\n')


MINIMAL_PATH = f'{BASE_RESULTS_PATH}/minimal_with_univariate.csv'
with open(MINIMAL_PATH, 'w') as f:
    f.write(f'período,função,carga,apps,feature,método,nmae\n')

BEST_K_PATH = f'{BASE_RESULTS_PATH}/best_k.csv'
with open(BEST_K_PATH, 'w') as f:
    f.write(f'período,função,Features,\n')



#   BEST_K_PATH = f'{BASE_RESULTS_PATH}/best_k.csv'
#   with open(BEST_K_PATH, 'w') as f:
#       f.write(f'período,função,Features,\n')


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
            # total

            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            for period in PERIODS:
                for name, func in functions:
                    x_filtered = pd.read_csv(
                        f'{AGG_DATASET_PATH}/X_{trace}_P-{period}_{name}_total.csv', index_col="TimeStamp")
                    y_filtered = y_dataset.loc[x_filtered.index]

                    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
                    # Realizar treinamento e teste com K-Fold  
                    for fold, (train_index, test_index) in enumerate(kf.split(x_filtered)):  
                        x_train, x_test = x_filtered.iloc[train_index], x_filtered.iloc[test_index]  
                        y_train, y_test = y_filtered.iloc[train_index], y_filtered.iloc[test_index]

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

                    k = 12
                    best_k = []
                    selectK = SelectKBest(f_regression, k=k)

                    selectK.set_output(transform="pandas")

                    minimal_dataset = selectK.fit_transform(x_filtered, y_filtered)
                    best_k.append(list(minimal_dataset.columns))

                    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
                    # Realizar treinamento e teste com K-Fold  
                    for fold, (train_index, test_index) in enumerate(kf.split(minimal_dataset)):  
                        x_train, x_test = minimal_dataset.iloc[train_index], minimal_dataset.iloc[test_index]  
                        y_train, y_test = y_filtered.iloc[train_index], y_filtered.iloc[test_index]

                    regression_tree_regressor = DecisionTreeRegressor()
                    regression_tree_regressor.fit(minimal_dataset, y_train)

                    random_forest_regressor = RandomForestRegressor(
                        n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                    random_forest_regressor.fit(minimal_dataset, y_train)

                    with open(MINIMAL_PATH, 'a') as f:
                        f.write(
                            f'{period},{name},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                        f.write(
                            f'{period},{name},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')

                    print(f'univariate {period},{name},{trace_load},{trace_apps},{y_metric},RF')

                    with open(BEST_K_PATH, 'a') as f:
                        f.write(f'{period},{name},{best_k},\n')