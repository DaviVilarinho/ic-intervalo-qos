import dataset_management
import experiment
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import global_variables_experiment
from sklearn.model_selection import train_test_split
import warnings
from os import makedirs
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from cli_parser import parse_strategy_and_sparsing_factor

warnings.filterwarnings('ignore')


def main(strategy, sparsing_factor, destination, random_state=42):
    results_path = global_variables_experiment.get_base_results_path(
        'replication/univariate')
    table_iv_columns = ['load_pattern', 'exp_type', 'regression_method',
                        'trace_family', 'y_metric', 'nmae', 'training_time']
    original_results = pd.DataFrame({
        'trace_family': ['VoD', 'KV'] * 8,
        'y_metric': ['DispFrames', 'ReadsAvg'] * 8,
        'regression_method': ['reg_tree', 'random_forest'] * 8,
        'load_pattern': (['PeriodicLoad'] * 8) + (['FlashcrowdLoad'] * 8),
        'exp_type': ['SingleApp', 'BothApps'] * 8,
        'nmae': [.14, .02, .11, .02, .12, .04, .11, .04, .11, .02, .09, .02, .1, .04, .06, .06],
        'training_time': [1, 1, 40, 32, 1, 1, 38, 30, 1, 1, 37, 16, 1, 1, 24, 11]
    }
    )

    results_entry_columns = [
        'k_best', 'sparsing_factor', 'random_state'] + table_iv_columns
    results = pd.DataFrame(columns=results_entry_columns)
    for trace_family, traces in global_variables_experiment.TRACES.items():
        for trace in traces:
            for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
                x, y = dataset_management.parse_traces(
                    trace, y_metric, ['X_port.csv', 'X_cluster.csv', 'X_flow.csv'])
                x, y = strategy.sparse(x, y, sparsing_factor)
                for k in range(1, 20):
                    for regression_method in ['reg_tree', 'random_forest']:

                        selectK = SelectKBest(f_regression, k=k)
                        selectK.set_output(transform="pandas")
                        minimal_dataset = selectK.fit_transform(x, y)

                        univariate_experiment = experiment.run_experiment(
                            minimal_dataset, y, y_metric, random_state=random_state, regression_method=regression_method)

                        _, exp_type, load_pattern = trace.split('-')
                        results = pd.concat([
                            results,
                            pd.DataFrame([{'random_state': random_state,
                                            'trace_family': trace_family,
                                            'y_metric': y_metric,
                                            'sparsing_factor': sparsing_factor,
                                            'k_best': k,
                                            'regression_method': regression_method,
                                            'load_pattern': load_pattern,
                                            'exp_type': exp_type,
                                            'nmae': univariate_experiment[regression_method]['nmae'],
                                            'training_time': univariate_experiment[regression_method]['training_time']}])],
                            ignore_index=True)

    results_path = destination or results_path
    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/univariate_table_iv_entries.csv')
    keys = ['trace_family', 'y_metric',
            'regression_method', 'load_pattern', 'exp_type']
    results = pd.merge(results, original_results,
                       suffixes=['', '_stepwise'], on=keys)
    results['delta_to_original_nmae'] = results['nmae'] - \
        results['nmae_stepwise']
    results.to_csv(f'{results_path}/univariate_table_iv_compared.csv')


if __name__ == '__main__':
    main(*parse_strategy_and_sparsing_factor())
