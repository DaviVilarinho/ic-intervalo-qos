from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

IS_LOCAL = os.uname()[1] == "eclipse"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/univariate_experiment/{DATE}'

def new_minimal_prediction(k, load_pattern, exp_type, regression, metric, metricNmae, training_time):
    return pd.DataFrame([{"k_best": k,
            "LoadPattern": load_pattern,
            "ExpType": exp_type,
            "RegressionMethod": regression,
            f'{metric}Nmae': metricNmae,
            f'{metric}TrainingTime': training_time}])

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

NROWS = 8 if IS_LOCAL else None

def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()

results_path = f'{BASE_RESULTS_PATH}'
for paths in [results_path]:
    try:
        os.makedirs(paths)
    except FileExistsError:
        pass

y_metrics = {
    "VOD": ['DispFrames'],#, 'noAudioPlayed'],
    "KV": ["ReadsAvg"]#, "WritesAvg"]
}

# the table i want to create is:
## 
tables = {
        "ReadsAvg": pd.DataFrame(columns=["k_best", "LoadPattern", "ExpType", "RegressionMethod", "ReadsAvgNmae", "ReadsAvgTrainingTime"]),
        "DispFrames": pd.DataFrame(columns=["k_best", "LoadPattern", "ExpType", "RegressionMethod", "DispFramesNmae", "DispFramesTrainingTime"]),
}
best_k = []



for trace_family, traces in traces.items():
    for trace in traces:
        for y_metric in y_metrics[trace_family]:
            for k in range(1,16+1):
                ## minimal
                trace_split = trace.split('-')

                x_files = ['X_cluster.csv', 'X_flow.csv', 'X_port.csv']

                x_trace = pd.DataFrame()

                for x_file in x_files:
                    x_trace_per_dataset = pd.read_csv(f'{PASQUINIS_PATH}/{trace}/{x_file}', 
                                            header=0, index_col=0, low_memory=True, nrows=NROWS).apply(pd.to_numeric, errors='coerce').fillna(0)
                    if len(x_trace.columns) != 0:
                        x_trace.merge(x_trace_per_dataset, how="inner",
                            on="TimeStamp", copy=False)
                    else:
                        x_trace = x_trace_per_dataset

                y_dataframe = pd.read_csv(f'{PASQUINIS_PATH}/{trace}/Y.csv', header=0, nrows=NROWS, index_col=0, usecols=['TimeStamp', y_metric], low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

                selectK = SelectKBest(f_regression, k=k)
                
                selectK.set_output(transform="pandas")
                minimal_dataset = selectK.fit_transform(x_trace, y_dataframe)
                
                best_k.append(list(minimal_dataset.columns))

                x_minimal_train, x_minimal_test, y_minimal_train, y_minimal_test = train_test_split(minimal_dataset, y_dataframe, test_size=0.3, random_state=42)

                regression_tree = DecisionTreeRegressor() 

                training_minimal_reg_tree_time = time.time()
                regression_tree.fit(x_minimal_train, y_minimal_train)
                training_minimal_reg_tree_time = time.time() - training_minimal_reg_tree_time

                pred_reg_tree_minimal = regression_tree.predict(x_minimal_test)
                
                tables[y_metric] = pd.concat([tables[y_metric], new_minimal_prediction(k, trace_split[2], trace_split[1], "RegTree", y_metric , nmae(pred_reg_tree_minimal, y_minimal_test[y_metric]), training_minimal_reg_tree_time)], ignore_index=True)

                random_forest_regressor = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)

                random_forest_minimal_training_time = time.time()
                random_forest_regressor.fit(x_minimal_train, y_minimal_train)
                random_forest_minimal_training_time = time.time() - random_forest_minimal_training_time

                pred_random_forest_minimal = random_forest_regressor.predict(x_minimal_test)

                tables[y_metric] = pd.concat([tables[y_metric],new_minimal_prediction(k, trace_split[2],trace_split[1], "RandomForest", y_metric , nmae(pred_random_forest_minimal, y_minimal_test[y_metric]), random_forest_minimal_training_time)], ignore_index=True)

table_iv = pd.merge(tables["ReadsAvg"], tables["DispFrames"], on=["k_best", "LoadPattern", "ExpType", "RegressionMethod"]).sort_values(by=['LoadPattern', 'ExpType', 'k_best'])

def make_original_table_result(load_pattern, exptype, regression_method, video_frame_rate_nmae, video_frame_rate_training_time, read_resp_nmae, read_resp_training_time):
    return {'LoadPattern': load_pattern,
            'ExpType': exptype,
            'RegressionMethod': regression_method,
            'ReadsAvgNmae': read_resp_nmae,
            'ReadsAvgTrainingTime': read_resp_training_time,
            'DispFramesNmae': video_frame_rate_nmae,
            'DispFramesTrainingTime': video_frame_rate_training_time}
    
original_minimal_results = pd.DataFrame([
    make_original_table_result('PeriodicLoad', 'SingleApp', 'RegTree', 0.14, 1, .02, 1),
    make_original_table_result('PeriodicLoad', 'BothApps', 'RandomForest', 0.11, 40, .02, 32),
    make_original_table_result('PeriodicLoad', 'SingleApp', 'RegTree', 0.12, 1, .02, 1),
    make_original_table_result('PeriodicLoad', 'BothApps', 'RandomForest', 0.11, 38, .02, 30),
    make_original_table_result('FlashCrowdLoad', 'SingleApp', 'RegTree', 0.11, 1, .02, 1),
    make_original_table_result('FlashCrowdLoad', 'BothApps', 'RandomForest', 0.09, 37, .02, 16),
    make_original_table_result('FlashCrowdLoad', 'SingleApp', 'RegTree', 0.10, 1, .02, 1),
    make_original_table_result('FlashCrowdLoad', 'BothApps', 'RandomForest', 0.06, 24, .02, 11),
])

table_iv = pd.merge(table_iv, original_minimal_results, on=["LoadPattern", "ExpType", "RegressionMethod"], suffixes=['', 'Original'])
table_iv['DeltaToOriginalReadsAvgNmae'] = table_iv['ReadsAvgNmae'] - table_iv['ReadsAvgNmaeOriginal']
table_iv['DeltaToOriginalDispFramesNmae'] = table_iv['DispFramesNmae'] - table_iv['DispFramesNmaeOriginal']

table_iv.to_csv(f'{BASE_RESULTS_PATH}/table_iv.csv')
with open(f'{BASE_RESULTS_PATH}/best_k.csv', 'w') as best_k_file:
    for each_k in best_k:
        best_k_file.write(f'{str(each_k)}\n')
