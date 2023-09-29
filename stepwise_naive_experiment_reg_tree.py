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

x_files = ['X_cluster.csv', 'X_flow.csv', 'X_port.csv']

x_trace = pd.DataFrame()

for x_file in x_files:
    read_dataset = pd.read_csv(f'{VOD_SINGLEAPP_PERIODIC_LOAD_PATH}/{x_file}', 
                              header=0, index_col=0, low_memory=True, nrows=NROWS).apply(pd.to_numeric, errors='coerce').fillna(0)
    if len(x_trace.columns) != 0:
        x_trace.merge(read_dataset, how="inner",
            on="TimeStamp", copy=False)
    else:
        x_trace = read_dataset

#minimal dataset creation
import warnings 

warnings.filterwarnings('ignore')

x_trace_corrwith_y_metric = abs(x_trace.corrwith(y_dataset[Y_METRIC]))
x_trace_corrwith_y_metric.fillna(0, inplace=True)
x_trace_corrwith_y_metric.sort_values(inplace=True, ascending=False)

time_to_build_minimal_dataset = time.time()
#training only with reg tree
## stepwise selection algorithm
old_reg_tree_nmae = 1
x_trace_minimal_reg_tree = pd.DataFrame()
while True:
    nmae_appending_feature_to_the_combination = {feature: 1 for feature in list(filter(lambda f: f not in x_trace_minimal_reg_tree.index, x_trace_corrwith_y_metric.index))}
    for feature in x_trace_corrwith_y_metric.index:
        x_trace_minimal_reg_tree[feature] = x_trace[[feature]].copy()

        X_train_minimal , X_test_minimal, y_train_minimal, y_test_minimal = train_test_split(x_trace_minimal_reg_tree, y_dataset, test_size=0.7, random_state=42)

        regression_tree_minimal = DecisionTreeRegressor() 
        regression_tree_minimal.fit(X_train_minimal, y_train_minimal)
        pred_reg_tree_minimal = regression_tree_minimal.predict(X_test_minimal)

        nmae_appending_feature_to_the_combination[feature] = nmae(pred_reg_tree_minimal, y_test_minimal[Y_METRIC])

        x_trace_minimal_reg_tree.drop([feature], axis=1, inplace=True)

    lowest_nmae_from_appending = min(nmae_appending_feature_to_the_combination, key=nmae_appending_feature_to_the_combination.get)
    if nmae_appending_feature_to_the_combination[lowest_nmae_from_appending] < old_reg_tree_nmae:
        break

    print(f'Appending {lowest_nmae_from_appending} because the NMAE with it {nmae_appending_feature_to_the_combination[lowest_nmae_from_appending]} < {old_reg_tree_nmae}.')
    old_reg_tree_nmae = nmae_appending_feature_to_the_combination[lowest_nmae_from_appending]
    x_trace_minimal_reg_tree[lowest_nmae_from_appending] = x_trace[[lowest_nmae_from_appending]].copy()


## end of  stepwise selection
time_to_build_minimal_dataset = time.time() - time_to_build_minimal_dataset

print(len(x_trace_minimal_reg_tree.columns))
print(x_trace_minimal_reg_tree.columns)

X_train_minimal , X_test_minimal, y_train_minimal, y_test_minimal = train_test_split(x_trace_minimal_reg_tree, y_dataset, test_size=0.7, random_state=42)

regression_tree_minimal = DecisionTreeRegressor() 

tempo_reg_tree_minimal = time.time()
regression_tree_minimal.fit(X_train_minimal, y_train_minimal)
tempo_reg_tree_minimal = time.time() - tempo_reg_tree_minimal


pred_reg_tree_minimal = regression_tree_minimal.predict(X_test_minimal)

new_nmae_reg_tree = nmae(pred_reg_tree_minimal, y_test_minimal[Y_METRIC])

print(f'MINIMAL NMAE REG_TREE {new_nmae_reg_tree}\n TRAINING TIME MINIMAL REG_TREE {tempo_reg_tree_minimal}')

random_forest_minimal = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)

tempo_random_forest_minimal = time.time()
random_forest_minimal.fit(X_train_minimal, y_train_minimal)
tempo_random_forest_minimal = time.time() - tempo_random_forest_minimal

pred_random_forest_minimal = random_forest_minimal.predict(X_test_minimal)

new_nmae_random_forest = nmae(pred_random_forest_minimal, y_test_minimal[Y_METRIC])

print(f'MINIMAL NMAE RANDOM_FOREST {new_nmae_random_forest}\n TRAINING TIME MINIMAL RANDOM FOREST {tempo_random_forest_minimal}')

