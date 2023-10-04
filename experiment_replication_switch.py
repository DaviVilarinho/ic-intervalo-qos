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
import joblib
from datetime import datetime
import concurrent
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "../"
PASQUINIS_PATH = BASE_DIR + "traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'./resultados_pre_pesquisa/{DATE}'
MODELS_DIR = "models/"
MODELS_PATH_PREFIX = f'models/{DATE}_'

TRACES=[
    #"KV-BothApps-FlashcrowdLoad",
    #"KV-BothApps-PeriodicLoad",
    #"KV-SingleApp-FlashcrowdLoad",
    #"KV-SingleApp-PeriodicLoad",
    #"VoD-BothApps-FlashcrowdLoad",
    #"VoD-BothApps-PeriodicLoad",
    #"VoD-SingleApp-FlashcrowdLoad",
    "VoD-SingleApp-PeriodicLoad"]

NROWS = None
PERSIST = False
Y_METRIC = 'DispFrames'
VOD_SINGLEAPP_PERIODIC_LOAD_PATH = f'{PASQUINIS_PATH}/VoD-SingleApp-PeriodicLoad'

def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()

if PERSIST:
    results_path = f'{BASE_RESULTS_PATH}'
    try:
        os.makedirs(results_path)
        os.makedirs(MODELS_DIR)
    except FileExistsError:
        print("Já criado diretório...")

y_dataset = pd.read_csv(f'{VOD_SINGLEAPP_PERIODIC_LOAD_PATH}/Y.csv', header=0, nrows=NROWS, index_col=0, usecols=['TimeStamp', 'DispFrames'], low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

x_files = ['X_port.csv']

x_trace = pd.DataFrame()

for x_file in x_files:
    read_dataset = pd.read_csv(f'{VOD_SINGLEAPP_PERIODIC_LOAD_PATH}/{x_file}', 
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

print('switch,regression_tree_nmae,time_to_train_regression_tree_s,random_forest_nmae,time_to_train_random_forest_s,')
for switch in per_switch_traces.keys():
    x_train_trace_per_switch , x_test_per_switch, y_train_per_switch, y_test_per_switch = train_test_split(per_switch_traces[switch], y_dataset, test_size=0.7, random_state=42)

    regression_tree_per_switch = DecisionTreeRegressor()

    tempo_reg_tree_per_switch = time.time()
    regression_tree_per_switch.fit(x_train_trace_per_switch, y_train_per_switch)
    tempo_reg_tree_per_switch = time.time() - tempo_reg_tree_per_switch

    pred_reg_tree_per_switch = regression_tree_per_switch.predict(x_test_per_switch)
    new_nmae_reg_tree = nmae(pred_reg_tree_per_switch, y_test_per_switch[Y_METRIC])


    random_forest_per_switch = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)

    tempo_random_forest_per_switch = time.time()
    random_forest_per_switch.fit(x_train_trace_per_switch, y_train_per_switch)
    tempo_random_forest_per_switch = time.time() - tempo_random_forest_per_switch

    pred_random_forest_per_switch = random_forest_per_switch.predict(x_test_per_switch)
    new_nmae_random_forest = nmae(pred_random_forest_per_switch, y_test_per_switch[Y_METRIC])

    print(f'{switch},{new_nmae_reg_tree},{tempo_reg_tree_per_switch},{new_nmae_random_forest},{tempo_random_forest_per_switch},')
