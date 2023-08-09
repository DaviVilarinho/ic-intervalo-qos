from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime

BASE_DIR = "../"
PASQUINIS_PATH = BASE_DIR + "traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'./resultados_pre_pesquisa/{DATE}'

TRACES=[
    #"KV-BothApps-FlashcrowdLoad",
    #"KV-BothApps-PeriodicLoad",
    #"KV-SingleApp-FlashcrowdLoad",
    #"KV-SingleApp-PeriodicLoad",
    #"VoD-BothApps-FlashcrowdLoad",
    #"VoD-BothApps-PeriodicLoad",
    #"VoD-SingleApp-FlashcrowdLoad",
    "VoD-SingleApp-PeriodicLoad"]

def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()

def get_nmae_random_forest_regression_tree(X, Y) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)
    
    regression_tree = DecisionTreeRegressor() # a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations. 
    regression_tree.fit(X_train, y_train)

    regr_random_forest = RandomForestRegressor(n_estimators=120, random_state=0)
    regr_random_forest.fit(X_train, y_train)

    y_random_forest = regr_random_forest.predict(X_test)
    y_reg_tree = regression_tree.predict(X_test)
    
    return (nmae(y_random_forest, y_test), nmae(y_reg_tree, y_test))


def read_traces(traces, NROWS=None):
    csvs = {}

    for root, _, files in os.walk(traces):
        for file in files:
            if file.find("csv") >= 0:
                csvs[file] = pd.read_csv(f'{root}/{file}', nrows=NROWS)

    return csvs

def cria_x_y(csvs_map: dict):
    return (pd.concat([csvs_map["X_flow.csv"], csvs_map["X_cluster.csv"], csvs_map["X_port.csv"]], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0),
            csvs_map["Y.csv"].apply(pd.to_numeric, errors='coerce').fillna(0))

def filter_correlation(x, y, correlation_min):
    if correlation_min is None:
        return x
    correlation = x.apply(lambda feature: abs(feature.corr(y)))
    return x[correlation[correlation > correlation_min].index]
    

def main():
    NROWS = 400
    VOD_SINGLEAPP_PERIODIC_LOAD = read_traces(f'{PASQUINIS_PATH}/VoD-SingleApp-PeriodicLoad', NROWS)

    results_path = f'{BASE_RESULTS_PATH}'
    try:
        os.mkdir(results_path)
    except FileExistsError:
        print("Já criado diretório...")

    video_results_path = results_path + '/' + 'resultado_video.txt'
    with open(video_results_path, 'w') as logger:
        logger.write('case, nmae_random_forest, nmae_regression_tree\n')

    x_trace, y_dataset = cria_x_y(VOD_SINGLEAPP_PERIODIC_LOAD)

    casos_de_teste = [('sem-pre-processamento', None), 
                      ('correlacao-0-4', 0.4),
                      ('correlacao-0-2', 0.2),
                      ('correlacao-0-1', 0.1)]

    for nome_teste, correlation in casos_de_teste:
        method_nmae_dict = get_nmae_random_forest_regression_tree(filter_correlation(x_trace, y_dataset['DispFrames'], correlation), y_dataset['DispFrames'])

        with open(video_results_path, 'a+') as logger:
            rand, reg = method_nmae_dict
            logger.write(f'{nome_teste}, {rand}, {reg}\n')
    #['noAudioPlayed']

if __name__ == "__main__":
    main()
