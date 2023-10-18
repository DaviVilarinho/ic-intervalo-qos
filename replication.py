from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
BASE_RESULTS_PATH = f'{"." if not IS_LOCAL else "/tmp"}/replication/{DATE}'

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

NROWS = 100

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

for trace_family, traces in traces.items():
    for trace in traces:
        for y_metric in y_metrics[trace_family]:
            # smallest first
            ## per switch

            y_dataset = pd.read_csv(f'{PASQUINIS_PATH}/{trace}/Y.csv', header=0, nrows=NROWS, index_col=0, usecols=['TimeStamp', y_metric], low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

            x_files = ['X_port.csv']

            x_trace = pd.DataFrame()

            for x_file in x_files:
                read_dataset = pd.read_csv(f'{PASQUINIS_PATH}/{trace}/{x_file}',
                                        header=0, index_col=0, low_memory=True, nrows=NROWS).apply(pd.to_numeric, errors='coerce').fillna(0)
                if len(x_trace.columns) != 0:
                    x_trace.merge(read_dataset, how="inner",
                        on="TimeStamp", copy=False)
                else:
                    x_trace = read_dataset

            switch_ports = {
                "SWC1": [0,1,2,3,4],
                "SWC2": [5,6,7,8,9],
                "SWC3": [10,11,12,13,14],
                "SWC4": [15,16,17,18,19],
                "SWA1": [20,21],
                "SWA2": [22,23],
                "SWA3": [24,25,26],
                "SWA4": [27,28],
                "SWA5": [29,30],
                "SWA6": [31,32,33],
                "SWB1": [34,35,36],
                "SWB2": [37,38],
                "SWB3": [39,40,41],
                "SWB4": [42,43]
            }

            switch_from_port = {f'{port}': switch for switch, ports in switch_ports.items() for port in ports}

            per_switch_traces = {switch: pd.DataFrame() for switch in switch_ports.keys()}

            for feature in x_trace.columns:
                port = feature.split('_')[0]
                switch = switch_from_port[port]
                per_switch_traces[switch] = x_trace[[feature]].copy()

            per_switch_file = f'{BASE_RESULTS_PATH}/{trace}_{y_metric}_per_switch.csv'
            with open(per_switch_file, 'w') as f:
                f.write(f'switch,regression_tree_{y_metric}_nmae,time_to_train_regression_tree_s,random_forest_{y_metric}_nmae,time_to_train_random_forest_s,\n')

            for switch in per_switch_traces.keys():
                x_train_trace_per_switch , x_test_per_switch, y_train_per_switch, y_test_per_switch = train_test_split(per_switch_traces[switch], y_dataset, test_size=0.7, random_state=42)

                regression_tree_per_switch = DecisionTreeRegressor()

                time_reg_tree_per_switch = time.time()
                regression_tree_per_switch.fit(x_train_trace_per_switch, y_train_per_switch)
                time_reg_tree_per_switch = time.time() - time_reg_tree_per_switch

                pred_reg_tree_per_switch = regression_tree_per_switch.predict(x_test_per_switch)
                new_nmae_reg_tree = nmae(pred_reg_tree_per_switch, y_test_per_switch[y_metric])


                random_forest_per_switch = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)

                time_random_forest_per_switch = time.time()
                random_forest_per_switch.fit(x_train_trace_per_switch, y_train_per_switch)
                time_random_forest_per_switch = time.time() - time_random_forest_per_switch

                pred_random_forest_per_switch = random_forest_per_switch.predict(x_test_per_switch)
                new_nmae_random_forest = nmae(pred_random_forest_per_switch, y_test_per_switch[y_metric])

                with open(per_switch_file, 'a') as f:
                    f.write(f'{switch},{new_nmae_reg_tree},{time_reg_tree_per_switch},{new_nmae_random_forest},{time_random_forest_per_switch},\n')


            ## per flow e per port

            for x_file in ['X_flow.csv', 'X_port.csv']:
                per_dataset_file = f'{BASE_RESULTS_PATH}/{trace}_{y_metric}_per_{x_file.split(".")[0]}.csv'
                with open(per_dataset_file, 'w') as f:
                    f.write(f'regression_tree_{y_metric}_{x_file}_nmae,time_to_train_regression_tree_s,random_forest_{y_metric}_{x_file}_nmae,time_to_train_random_forest_s,\n')

                x_trace_per_dataset = pd.DataFrame()

                x_trace_per_dataset = pd.read_csv(f'{PASQUINIS_PATH}/{trace}/{x_file}', 
                                        header=0, index_col=0, low_memory=True, nrows=NROWS).apply(pd.to_numeric, errors='coerce').fillna(0)

                x_train_flow, x_test_flow, y_train_flow, y_test_per_dataset = train_test_split(x_trace_per_dataset, y_dataset, test_size=0.7, random_state=42)

                regression_tree_flow = DecisionTreeRegressor() 

                time_reg_tree_per_dataset = time.time()
                regression_tree_flow.fit(x_train_flow, y_train_flow)
                time_reg_tree_per_dataset = time.time() - time_reg_tree_per_dataset

                pred_reg_tree_per_dataset = regression_tree_flow.predict(x_test_flow)

                regr_random_forest_per_dataset = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)

                time_random_forest_per_dataset = time.time()
                regr_random_forest_per_dataset.fit(x_train_flow, y_train_flow)
                time_random_forest_per_dataset = time.time() - time_random_forest_per_dataset

                pred_random_forest_flow = regr_random_forest_per_dataset.predict(x_test_flow)
                with open(per_dataset_file, 'a') as f:
                    f.write(f'{nmae(pred_reg_tree_per_dataset, y_test_per_dataset[y_metric])},{time_reg_tree_per_dataset}, {nmae(pred_random_forest_flow, y_test_per_dataset[y_metric])},{time_random_forest_per_dataset},\n')

            # larger after

            ## total

            total_X_file_path = f'{BASE_RESULTS_PATH}/{trace}_{y_metric}_X.csv'
            with open(total_X_file_path, 'w') as f:
                f.write(f'regression_tree_{y_metric}_X_nmae,time_to_train_regression_tree_s,random_forest_{y_metric}_X_nmae,time_to_train_random_forest_s,\n')
            x_files = ['X_cluster.csv', 'X_flow.csv', 'X_port.csv']

            x_trace = pd.DataFrame()

            for x_file in x_files:
                x_trace_per_dataset = pd.read_csv(f'{PASQUINIS_PATH}/{trace}/{x_file}', 
                                        header=0, index_col=0, low_memory=True, nrows=NROWS).apply(pd.to_numeric, errors='coerce').fillna(0)
                if len(x_trace.columns) != 0:
                    x_trace.merge(read_dataset, how="inner",
                        on="TimeStamp", copy=False)
                else:
                    x_trace = read_dataset

            x_total_train, x_total_test, y_total_train, y_total_test = train_test_split(x_trace, y_dataset, test_size=0.7, random_state=42)

            regression_tree = DecisionTreeRegressor() 

            training_total_reg_tree_time = time.time()
            regression_tree.fit(x_total_train, y_total_train)
            training_total_reg_tree_time = time.time() - training_total_reg_tree_time

            pred_reg_tree_total = regression_tree.predict(x_total_test)

            random_forest_regressor = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)

            random_forest_total_X_training_time = time.time()
            random_forest_regressor.fit(x_total_train, y_total_train)
            random_forest_total_X_training_time = time.time() - random_forest_total_X_training_time

            pred_random_forest_total = random_forest_regressor.predict(x_total_test)

            with open(total_X_file_path, 'a') as f:
                f.write(f'{nmae(pred_reg_tree_total, y_total_test[y_metric])},{training_total_reg_tree_time}, {nmae(pred_random_forest_total, y_test_per_dataset[y_metric])},{random_forest_total_X_training_time},\n')
            ## minimal

            ### reg tree

            ### random forest
