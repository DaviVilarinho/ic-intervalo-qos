import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime
import warnings
from dataset_management import parse_traces
from sklearn.model_selection import KFold

K_FOLD = 3
warnings.filterwarnings('ignore')

IS_LOCAL = False
RANDOM_STATE = 42
EXPERIMENT = "experiment_4_7200samples"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/{EXPERIMENT}/{DATE}'

DISTRIB_DATASET_PATH = "distrib_function_periodic_dataset"
AGG_DATASET_PATH = "agg_function_periodic_dataset"

traces = {
    "VOD": [
        "VoD-BothApps-FlashcrowdLoad",
        "VoD-BothApps-PeriodicLoad",
        "VoD-SingleApp-FlashcrowdLoad",
        "VoD-SingleApp-PeriodicLoad"],
}

functions = [
    ('média', np.mean),
    ('máximo', np.max),
    ('mínimo', np.min)
]

NROWS = None 

RANDOM_FOREST_TREES = 120


def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()


results_path = f'{BASE_RESULTS_PATH}'
for paths in [results_path]:
    try:
        os.makedirs(paths)
    except FileExistsError:
        pass

MINIMAL_PATH = f'{BASE_RESULTS_PATH}/minimal_with_univariate.csv'
with open(MINIMAL_PATH, 'w') as f:
    f.write(f'experimento,período,carga,apps,feature,método,nmae,função\n')

BEST_K_PATH = f'{BASE_RESULTS_PATH}/best_k.csv'
with open(BEST_K_PATH, 'w') as f:
    f.write(f'experimento,período,Features,função\n')

y_metrics = {
    "VOD": ['DispFrames',]  # 'noAudioPlayed'],
}

TOTAL_X_FILE_PATH = f'{BASE_RESULTS_PATH}/total_X.csv'
with open(TOTAL_X_FILE_PATH, 'w') as f:
    f.write(f'experimento,período,carga,apps,feature,método,nmae,função\n')


PERIODS = [2, 4, 8, 16, 32, 64, 128, 256]
PERIODS.reverse()

def filter_periodic(x, y, period: int):
    mask = (x.index % period == 0)
    x_filtered = x.loc[mask]
    y_filtered = y.loc[mask]

    return x_filtered, y_filtered

for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            for period in PERIODS:
                experiment = 3
                if period >= 8:
                    x_filtered = pd.read_csv(
                        f'{DISTRIB_DATASET_PATH}/X_{trace}_P-{period}_total.csv', index_col="TimeStamp")
                    y_filtered = y_dataset.loc[x_filtered.index]
                    x_filtered = x_filtered.fillna(0)
                    y_filtered = y_filtered.fillna(0)

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
                            f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},raw\n')
                        f.write(
                            f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},raw\n')
                    print(f'total X {experiment} {period},{trace_load},{trace_apps},{y_metric}')

                    
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
                    regression_tree_regressor.fit(x_train, y_train)

                    random_forest_regressor = RandomForestRegressor(
                        n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                    random_forest_regressor.fit(x_train, y_train)

                    with open(MINIMAL_PATH, 'a') as f:
                        f.write(
                            f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},raw\n')
                        f.write(
                            f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},raw\n')

                    print(f'univariate {experiment} {period},{trace_load},{trace_apps},{y_metric},RF')

                    with open(BEST_K_PATH, 'a') as f:
                        f.write(f'{period},{best_k},raw\n')
                
                experiment = 2
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
                            f'{experiment},{period},{name},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},{name}\n')
                        f.write(
                            f'{experiment},{period},{name},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},{name}\n')
                    print(f'total X {experiment}/{name} {period},{
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
                    regression_tree_regressor.fit(x_train, y_train)

                    random_forest_regressor = RandomForestRegressor(
                        n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                    random_forest_regressor.fit(x_train, y_train)

                    with open(MINIMAL_PATH, 'a') as f:
                        f.write(
                            f'{experiment},{period},{name},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},{name}\n')
                        f.write(
                            f'{experiment},{period},{name},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},{name}\n')

                    print(f'univariate {experiment}/{name} {period},{trace_load},{trace_apps},{y_metric},RF')

                    with open(BEST_K_PATH, 'a') as f:
                        f.write(f'{experiment},{period},{name},{best_k},{name}\n')
                
                experiment = 1

                x_filtered, y_filtered = filter_periodic(x_trace, y_dataset, period)

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
                        f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},raw\n')
                    f.write(
                        f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},raw\n')
                print(f'total X {experiment} {period},{trace_load},{trace_apps},{y_metric},RF')


                # minimal
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
                regression_tree_regressor.fit(x_train, y_train)

                random_forest_regressor = RandomForestRegressor(
                    n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                random_forest_regressor.fit(x_train, y_train)


                with open(MINIMAL_PATH, 'a') as f:
                    f.write(
                        f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RT,{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},raw\n')
                    f.write(
                        f'{experiment},{period},{trace_load},{trace_apps},{y_metric},RF,{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},raw\n')

                print(f'univariate {experiment} {period},{trace_load},{trace_apps},{y_metric},RF')

                with open(BEST_K_PATH, 'a') as f:
                    f.write(f'{experiment},{period},{best_k},raw\n')

