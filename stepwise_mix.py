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

IS_LOCAL = os.uname()[1].split('-').pop(0) == "ST"
RANDOM_STATE = 42 if IS_LOCAL else None
EXPERIMENT = "stepwise_mix"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/{EXPERIMENT}/{DATE}'

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

MINIMAL_PATH = f'{BASE_RESULTS_PATH}/minimal_with_univariate_and_stepwise.csv'
with open(MINIMAL_PATH, 'w') as f:
    f.write(f'carga,apps,feature,método, mét. stepwise,nmae\n')
BEST_K_PATH = f'{BASE_RESULTS_PATH}/features_minimized.csv' 

def stepwise_selection(x_trace, y_dataset, y_metric, regressor):
    old_reg_tree_nmae = 1
    x_trace_minimal = pd.DataFrame()
    while True:
        features_available = list(
            filter(lambda f: f not in x_trace_minimal.columns, x_trace.columns))
        if len(features_available) == 0:
            break
        nmae_appending_feature_to_the_combination = {
            feature: 1 for feature in features_available}
        for feature in features_available:
            x_trace_minimal[feature] = x_trace[[feature]].copy()

            x_train_minimal, x_test_minimal, y_train_minimal, y_test_minimal = train_test_split(
                x_trace_minimal, y_dataset, test_size=0.3, random_state=42)

            regressor.fit(x_train_minimal, y_train_minimal)
            pred_reg_tree_minimal = regressor.predict(x_test_minimal)

            nmae_appending_feature_to_the_combination[feature] = nmae(
                pred_reg_tree_minimal, y_test_minimal[y_metric])

            x_trace_minimal.drop([feature], axis=1, inplace=True)

        lowest_nmae_from_appending = min(
            nmae_appending_feature_to_the_combination, key=nmae_appending_feature_to_the_combination.get)
        if nmae_appending_feature_to_the_combination[lowest_nmae_from_appending] > old_reg_tree_nmae:
            break

        print(
            f'Appending {lowest_nmae_from_appending} because the NMAE with it is {nmae_appending_feature_to_the_combination[lowest_nmae_from_appending]} < {old_reg_tree_nmae} (old).')
        old_reg_tree_nmae = nmae_appending_feature_to_the_combination[lowest_nmae_from_appending]
        x_trace_minimal[lowest_nmae_from_appending] = x_trace[[
            lowest_nmae_from_appending]].copy()
    return x_trace_minimal

for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'])

            x_train, x_test, y_train, y_test = train_test_split(
                x_trace, y_dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE)


            # minimal
            k = 100
            best_k = []
            selectK = SelectKBest(f_regression, k=k)

            selectK.set_output(transform="pandas")

            minimal_dataset = selectK.fit_transform(x_trace, y_dataset)

            for regressor in [DecisionTreeRegressor(), RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)]:
                minimal_dataset = stepwise_selection(
                    minimal_dataset, y_dataset, y_metric, regressor)

                best_k.append(list(minimal_dataset.columns))

                x_train, x_test, y_train, y_test = train_test_split(
                    minimal_dataset, y_dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                regression_tree_regressor = DecisionTreeRegressor()
                regression_tree_regressor.fit(x_train, y_train)

                random_forest_regressor = RandomForestRegressor(
                    n_estimators=RANDOM_FOREST_TREES, random_state=RANDOM_STATE, n_jobs=-1)
                random_forest_regressor.fit(x_train, y_train)


                with open(MINIMAL_PATH, 'a') as f:
                    f.write(
                        f'{trace_load},{trace_apps},{y_metric},RT,{regressor},{nmae(regression_tree_regressor.predict(x_test), y_test[y_metric])},\n')
                    f.write(
                        f'{trace_load},{trace_apps},{y_metric},RF,{regressor},{nmae(random_forest_regressor.predict(x_test), y_test[y_metric])},\n')

                print(f'minimized {trace_load},{trace_apps},{y_metric},RF')

                with open(BEST_K_PATH, 'a') as f:
                    f.write(f'{best_k},\n')
