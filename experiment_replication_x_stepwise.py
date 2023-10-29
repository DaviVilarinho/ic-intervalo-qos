import dataset_management 
import experiment
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import global_variables_experiment
from sklearn.model_selection import train_test_split
import warnings
from os import makedirs
import pandas as pd

warnings.filterwarnings('ignore')
def stepwise_selection(x_trace, y_dataset, y_metric, regressor, random_state=42):
    old_nmae = 1
    x_trace_minimal = pd.DataFrame()
    while True:
        features_available = list(filter(lambda f: f not in x_trace_minimal.columns, x_trace.columns))
        if len(features_available) == 0:
            break
        nmae_appending_feature_to_the_combination = {feature: 1 for feature in features_available}
        for feature in features_available:
            x_trace_minimal[feature] = x_trace[[feature]].copy()

            x_train_minimal , x_test_minimal, y_train_minimal, y_test_minimal = train_test_split(x_trace_minimal, y_dataset, test_size=experiment.TEST_SIZE, random_state=random_state)

            regressor.fit(x_train_minimal, y_train_minimal[y_metric])
            pred_regressor = regressor.predict(x_test_minimal)

            nmae_appending_feature_to_the_combination[feature] = experiment.nmae(pred_regressor, y_test_minimal[y_metric])

            x_trace_minimal.drop([feature], axis=1, inplace=True)

        lowest_nmae_from_appending = min(nmae_appending_feature_to_the_combination, key=nmae_appending_feature_to_the_combination.get)
        if nmae_appending_feature_to_the_combination[lowest_nmae_from_appending] > old_nmae or (global_variables_experiment.IS_LOCAL and x_trace_minimal.shape[1] > 5):
            break

        print(f'Appending {lowest_nmae_from_appending} because the NMAE with it is {nmae_appending_feature_to_the_combination[lowest_nmae_from_appending]} < {old_nmae} (old).')
        old_nmae = nmae_appending_feature_to_the_combination[lowest_nmae_from_appending]
        x_trace_minimal[lowest_nmae_from_appending] = x_trace[[lowest_nmae_from_appending]].copy()
    return x_trace_minimal


def main():
    results_path = global_variables_experiment.get_base_results_path('replication/stepwise')
    table_iv_columns = ['load_pattern', 'exp_type', 'regression_method', 'trace_family', 'y_metric', 'nmae', 'training_time']
    original_results = pd.DataFrame({
        'trace_family': ['VoD', 'KV'] * 8,
        'y_metric': ['DispFrames', 'ReadsAvg'] * 8,
        'regression_method': ['reg_tree', 'random_forest'] * 8,
        'load_pattern': (['PeriodicLoad'] * 8) + (['FlashcrowdLoad'] * 8),
        'exp_type': ['SingleApp', 'BothApps'] * 8,
        'nmae': [.14, .02, .11, .02, .12, .04, .11, .04, .11, .02, .09, .02, .1, .04, .06, .06],
        'training_time': [1,1,40,32,1,1,38,30,1,1,37,16,1,1,24,11]
        }
    )

    results_entry_columns = ['random_state'] + table_iv_columns
    results = pd.DataFrame(columns=results_entry_columns)
    for trace_family, traces in global_variables_experiment.TRACES.items():
        for trace in traces:
            for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
                x,y = dataset_management.parse_traces(trace, y_metric, ['X_port.csv', 'X_cluster.csv', 'X_flow.csv'])
                for random_state in range(2):
                    for regression_method in ['reg_tree', 'random_forest']:
                        
                        minimal_dataset = stepwise_selection(x, y, y_metric, DecisionTreeRegressor() if regression_method == 'reg_tree' else RandomForestRegressor(n_estimators=experiment.RANDOM_FOREST_TREES, random_state=random_state, n_jobs=-1))

                        stepwise_experiment = experiment.run_experiment(minimal_dataset, y, y_metric, random_state=random_state, regression_method=regression_method)

                        _, exp_type, load_pattern = trace.split('-')
                        results = pd.concat([
                            results,
                            pd.DataFrame([{'random_state': random_state,
                                           'trace_family': trace_family,
                                           'y_metric': y_metric,
                                           'regression_method': regression_method,
                                           'load_pattern': load_pattern,
                                           'exp_type': exp_type,
                                           'nmae': stepwise_experiment[regression_method]['nmae'],
                                           'training_time': stepwise_experiment[regression_method]['training_time']}])], 
                            ignore_index=True)


    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/stepwise_table_iv_entries.csv')
    keys = ['trace_family', 'y_metric', 'regression_method', 'load_pattern', 'exp_type']
    results = results.groupby(keys).mean()
    results = pd.merge(results, original_results, suffixes=['', '_original'], on=keys)
    results['delta_to_original_nmae'] = results['nmae'] - results['nmae_original']
    results.to_csv(f'{results_path}/stepwise_table_iv_compared.csv')
    

if __name__ == '__main__':
    main()
