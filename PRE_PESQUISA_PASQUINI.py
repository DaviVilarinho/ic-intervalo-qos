from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error

BASE_DIR = "../"
PASQUINIS_PATH = BASE_DIR + "traces-netsoft-2017"
DATE = datetime.datetime.now().isoformat()
BASE_RESULTS_PATH = f'./resultados_pre_pesquisa/{DATE}'
RESULTS_VIDEO_PATH =  BASE_RESULTS_PATH + "/" + "resultado_linhas_video.txt"
RESULTS_AUDIO_PATH =  BASE_RESULTS_PATH + "/" + "resultado_linhas_audio.txt"
INTERVAL_VIDEO_PATH = BASE_RESULTS_PATH + "/" + "resultado_intervalos_video.txt"
INTERVAL_AUDIO_PATH = BASE_RESULTS_PATH + "/" + "resultado_intervalos_audio.txt"

TRACES=[
    #"KV-BothApps-FlashcrowdLoad",
    #"KV-BothApps-PeriodicLoad",
    #"KV-SingleApp-FlashcrowdLoad",
    #"KV-SingleApp-PeriodicLoad",
    #"VoD-BothApps-FlashcrowdLoad",
    #"VoD-BothApps-PeriodicLoad",
    #"VoD-SingleApp-FlashcrowdLoad",
    "VoD-SingleApp-PeriodicLoad"]

def start_results_file(path):
    try:
        os.mkdir(BASE_RESULTS_PATH)
    except FileExistsError:
        print("Já criado diretório...")
    with open(path, 'w') as logger:
        logger.write(f'parameter, predict_random_forest, regression_tree\n')

def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()

def get_nmae_and_predict_random_and_regression(X, Y) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)
    
    regression_tree = DecisionTreeRegressor(max_depth=2) # a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations. 
    regression_tree.fit(X_train, y_train)

    regr_random_forest = RandomForestRegressor(n_estimators=120, random_state=0)
    regr_random_forest.fit(X_train, y_train)

    y_random_forest = regr_random_forest.predict(X_test)
    y_reg_tree = regression_tree.predict(X_test)
    
    return (nmae(y_random_forest, y_test), nmae(y_reg_tree, y_test))

def save_results(path, key=None, random_forest=None, regression_tree=None):
    with open(path, 'a+') as logger:
        logger.write(f'{key}, {random_forest}, {regression_tree}\n')

def read_traces(traces, NROWS=None):
    csvs = {}

    for root, subfolders, files in os.walk(traces):
        for file in files:
            if file.find("csv") >= 0:
                csvs[file] = pd.read_csv(f'{root}/{file}', nrows=NROWS)

    return csvs

def cria_x_y(csvs_map: dict):
    return (pd.concat([csvs_map["X_flow.csv"], csvs_map["X_cluster.csv"], csvs_map["X_port.csv"]], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0),
            csvs_map["Y.csv"].apply(pd.to_numeric, errors='coerce').fillna(0))


def main_intervals():
    start_results_file(INTERVAL_VIDEO_PATH)
    start_results_file(INTERVAL_AUDIO_PATH)

    for interval in [256,128,64,32,16,8,4,2,1]:
        print(f'Analisando intervalo {interval}')
        VOD_SINGLEAPP_PERIODIC_LOAD = read_traces(f'{PASQUINIS_PATH}/VoD-SingleApp-PeriodicLoad')

        print(f'Cria x_y')
        X, y = cria_x_y(VOD_SINGLEAPP_PERIODIC_LOAD)
        
        print(f'Query para {interval}')
        X.query(f'TimeStamp % {interval} == 0', inplace=True)
        y.query(f'TimeStamp % {interval} == 0', inplace=True)

        print(f'Prevendo para vídeo com intervalo {interval}')
        random_forest_nmae, regression_tree_nmae = get_nmae_and_predict_random_and_regression(X, y['DispFrames'])
        print(f'Salvando para vídeo com intervalo {interval}')
        save_results(INTERVAL_VIDEO_PATH, key=interval, random_forest=random_forest_nmae, regression_tree=regression_tree_nmae)
        print(f'Prevendo para áudio com intervalo {interval}')
        random_forest_nmae, regression_tree_nmae = get_nmae_and_predict_random_and_regression(X, y['noAudioPlayed'])
        print(f'Salvando para áudio com intervalo {interval}')
        save_results(INTERVAL_AUDIO_PATH, key=interval, random_forest=random_forest_nmae, regression_tree=regression_tree_nmae)
         
def main_different_sizes():
    start_results_file(RESULTS_AUDIO_PATH)
    start_results_file(RESULTS_VIDEO_PATH)

    for NROWS in [None]:
        VOD_SINGLEAPP_PERIODIC_LOAD = read_traces(f'{PASQUINIS_PATH}/VoD-SingleApp-PeriodicLoad', NROWS)

        X, y = cria_x_y(VOD_SINGLEAPP_PERIODIC_LOAD)

        NROWS = 'todas_linhas'
        random_forest_nmae, regression_tree_nmae = get_nmae_and_predict_random_and_regression(X, y['DispFrames'])
        save_results(RESULTS_VIDEO_PATH, key=NROWS, random_forest=random_forest_nmae, regression_tree=regression_tree_nmae)
        random_forest_nmae, regression_tree_nmae = get_nmae_and_predict_random_and_regression(X, y['noAudioPlayed'])
        save_results(RESULTS_AUDIO_PATH, key=NROWS, random_forest=random_forest_nmae, regression_tree=regression_tree_nmae)

if __name__ == "__main__":
    main_different_sizes()
    #main_intervals()
