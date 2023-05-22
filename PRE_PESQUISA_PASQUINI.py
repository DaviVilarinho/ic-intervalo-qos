import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

BASE_DIR = "../"
PASQUINIS_PATH = BASE_DIR + "traces-netsoft-2017"
RESULTS_VIDEO_PATH = "./resultados_pre_pesquisa/resultado_video.txt"

TRACES=[
    #"KV-BothApps-FlashcrowdLoad",
    #"KV-BothApps-PeriodicLoad",
    #"KV-SingleApp-FlashcrowdLoad",
    #"KV-SingleApp-PeriodicLoad",
    #"VoD-BothApps-FlashcrowdLoad",
    #"VoD-BothApps-PeriodicLoad",
    #"VoD-SingleApp-FlashcrowdLoad",
    "VoD-SingleApp-PeriodicLoad"]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#NROWS = 7e3
def read_traces(traces, NROWS):
    csvs = {}

    for root, subfolders, files in os.walk(traces):
        for file in files:
            if file.find("csv") >= 0:
                csvs[file] = pd.read_csv(f'{root}/{file}', nrows=NROWS)

    return csvs

for NROWS in [1e3, 2e3, 4e3, 8e3, 16e3, 28950]:
    VOD_SINGLEAPP_PERIODIC_LOAD = read_traces(f'{PASQUINIS_PATH}/VoD-SingleApp-PeriodicLoad', NROWS)

    print(VOD_SINGLEAPP_PERIODIC_LOAD.keys())

    def cria_x_y(csvs_map: dict):
        return (pd.concat([csvs_map["X_flow.csv"], csvs_map["X_cluster.csv"], csvs_map["X_port.csv"]], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0),
                csvs_map["Y.csv"].apply(pd.to_numeric, errors='coerce').fillna(0))

    X, y = cria_x_y(VOD_SINGLEAPP_PERIODIC_LOAD)

    y_video = y['DispFrames']

    np.set_printoptions(threshold=sys.maxsize)

    X_video_train, X_video_test, y_video_train, y_video_test = train_test_split(X, y_video, test_size=0.7, random_state=42)

    regr_1 = DecisionTreeRegressor(max_depth=2) # a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations. 
    regr_1.fit(X_video_train, y_video_train)

    regr_random_forest = RandomForestRegressor(n_estimators=120, random_state=0)
    regr_random_forest.fit(X_video_train, y_video_train)

#y_1 = regr_1.predict(X_video_test)
    y_random_forest = regr_random_forest.predict(X_video_test)
    y_reg_tree = regr_1.predict(X_video_test)

    from sklearn.metrics import mean_squared_error
#print(mean_squared_error(y_video_test, y_1))
    with open(RESULTS_VIDEO_PATH, 'a+') as logger:
        logger.write(f'{NROWS}, {mean_squared_error(y_video_test, y_random_forest)}, {mean_squared_error(y_video_test, y_reg_tree)}\n')
    print(mean_squared_error(y_video_test, y_random_forest))


# e para áudio
#y_audio = y#[''] # o que considerar?
#X_audio_train, X_audio_test, y_audio_train, y_audio_test = train_test_split(X, y_audio, test_size=0.33, random_state=42)

#regr_random_forest.fit(X, y_audio)d
#y_audio_random_forest = regr_random_forest.predict(X_test)

#print(mean_squared_error(y_audio, y_audio_random_forest))
