from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
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

def nmae(y_pred, y_test):
    return abs(y_pred - y_test).mean() / y_test.mean()

def get_nmae_random_forest_regression_tree(X, Y) -> tuple:
    print("Splitando em testes")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)
    
    print("Treinando Decision tree")
    regression_tree = DecisionTreeRegressor() # a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations. 
    regression_tree.fit(X_train, y_train)

    joblib.dump(regression_tree, f'{MODELS_PATH_PREFIX}_model_regression-tree.sav')

    print("Treinando random forest")
    regr_random_forest = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    regr_random_forest.fit(X_train, y_train)

    joblib.dump(regr_random_forest, f'{MODELS_PATH_PREFIX}_model_random-forest.sav')

    y_random_forest = regr_random_forest.predict(X_test)
    y_reg_tree = regression_tree.predict(X_test)
    
    return (nmae(y_random_forest, y_test), nmae(y_reg_tree, y_test))

def filter_correlation(x, y, correlation_min):
    if correlation_min is None:
        return x
    correlation = x.apply(lambda feature: abs(feature.corr(y)))
    return x[correlation[correlation > correlation_min].index]
    

NROWS = 1000
Y_METRIC = 'DispFrames'
VOD_SINGLEAPP_PERIODIC_LOAD_PATH = f'{PASQUINIS_PATH}/VoD-SingleApp-PeriodicLoad'

def get_correlated_columns(x_file, group_column, y_dataset, correlation, Y_METRIC=Y_METRIC):
    column_dataset = pd.read_csv(f'{VOD_SINGLEAPP_PERIODIC_LOAD_PATH}/{x_file}', index_col=0, usecols=np.append(['TimeStamp'], group_column),nrows=NROWS).apply(pd.to_numeric, errors='coerce').fillna(0)
    for column in group_column:
        corr_to_y = abs(column_dataset[column].corr(y_dataset[Y_METRIC]))
        if corr_to_y > correlation:
            print(f'{column} abs corr to y {corr_to_y}, putting on x_trace')
        else:
            column_dataset = column_dataset.drop([column], axis=1)
    if column_dataset.isnull().values.any():
        print(column_dataset.columns.to_list())
    return column_dataset

def main():
    casos_de_teste = [#('sem-pre-processamento', None), 
                      ('correlacao-0-4', 0.3),
                      ('correlacao-0-2', 0.2),
                      ('correlacao-0-1', 0.1)]

    results_path = f'{BASE_RESULTS_PATH}'
    try:c
        os.makedirs(results_path)
        os.makedirs(MODELS_DIR)
    except FileExistsError:
        print("Já criado diretório...")


    video_results_path = results_path + '/' + 'resultado_video.txt'
    video_correlated_columns_path = results_path + '/' + f'columns_with_abscorr.txt'

    with open(video_results_path, 'w') as logger:
        logger.write(f'case_with_NROWS_{NROWS}, nmae_random_forest, nmae_regression_tree\n')

    with open(video_correlated_columns_path, 'w') as logger:
        logger.write(f'correlation_with_NROWS_{NROWS}\t\t columns\n')

    for nome_teste, correlation in casos_de_teste:

        y_dataset = pd.read_csv(f'{VOD_SINGLEAPP_PERIODIC_LOAD_PATH}/Y.csv', nrows=NROWS, index_col=0, usecols=['TimeStamp', 'DispFrames'], low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

        x_files = ['X_cluster.csv', 'X_flow.csv', 'X_port.csv']
        x_columns_by_file = {x_file: pd.read_csv(f'{VOD_SINGLEAPP_PERIODIC_LOAD_PATH}/{x_file}', index_col=0, nrows=0).columns.to_list() for x_file in x_files}

        x_trace = pd.DataFrame(columns=[col for _, x_cols_from_file in x_columns_by_file.items() for col in x_cols_from_file])

        for x_file, columns in x_columns_by_file.items():
            x_columns_by_file[x_file] = np.array_split(np.array(columns), 40)

        with concurrent.futures.ProcessPoolExecutor(8) as executor:
            for x_file, groups in x_columns_by_file.items():
                group_futures = [executor.submit(get_correlated_columns, x_file, group_column, y_dataset, correlation) for group_column in groups]
                for future_group_dataset in concurrent.futures.as_completed(group_futures):
                    group_dataset = future_group_dataset.result()
                    for column in group_dataset.columns.to_list():
                        x_trace[column] = group_dataset[column]

        x_trace = x_trace.dropna(axis=1, how='all')

        rand, reg = get_nmae_random_forest_regression_tree(x_trace, y_dataset['DispFrames'])

        with open(video_results_path, 'a+') as logger:
            logger.write(f'{nome_teste}, {rand}, {reg}\n')
        with open(video_correlated_columns_path, 'a+') as logger:
            logger.write(f'{correlation}\t\t {x_trace.columns.to_list()}\n')
        x_trace.to_csv(f'{results_path}/X_dataframe_min_correlation_{correlation}.csv')
        #['noAudioPlayed']

if __name__ == "__main__":
    main()
