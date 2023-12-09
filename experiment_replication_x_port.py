import dataset_management
import experiment
import global_variables_experiment
import warnings
from os import makedirs
import pandas as pd

from cli_parser import parse_strategy_and_sparsing_factor

warnings.filterwarnings('ignore')


def main(strategy, sparsing_factor, destination, random_state=42):
    results_path = global_variables_experiment.get_base_results_path(
        'replication/port')
    table_v_columns = ['load_pattern', 'exp_type', 'regression_method',
                       'trace_family', 'y_metric', 'nmae', 'training_time']
    original_results = pd.DataFrame({
        'trace_family': ['VoD', 'KV'] * 8,
        'y_metric': ['DispFrames', 'ReadsAvg'] * 8,
        'regression_method': ['reg_tree', 'random_forest'] * 8,
        'load_pattern': (['PeriodicLoad'] * 8) + (['FlashcrowdLoad'] * 8),
        'exp_type': ['SingleApp', 'BothApps'] * 8,
        'nmae': [.11, .03, .11, .02, .15, .05, .14, .04, .11, .02, .1, .02, .11, .04, .1, .03],
        'training_time': [16, 17, 4200, 4600, 14, 13, 3400, 3900, 85, 15, 3800, 2400, 15, 10, 3100, 7600]
    }
    )

    results_entry_columns = ['random_state'] + table_v_columns
    results = pd.DataFrame(columns=results_entry_columns)
    for trace_family, traces in global_variables_experiment.TRACES.items():
        for trace in traces:
            for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
                x, y = dataset_management.parse_traces(
                    trace, y_metric, ['X_port.csv'])
                x, y = strategy.sparse(x, y, sparsing_factor)

                port_experiment = experiment.run_experiment(
                    x, y, y_metric, random_state=random_state)

                _, exp_type, load_pattern = trace.split('-')
                for regression_method in ['reg_tree', 'random_forest']:
                    results = pd.concat([
                        results,
                        pd.DataFrame([{'random_state': random_state,
                                        'trace_family': trace_family,
                                        'y_metric': y_metric,
                                        'regression_method': regression_method,
                                        'load_pattern': load_pattern,
                                        'exp_type': exp_type,
                                        'nmae': port_experiment[regression_method]['nmae'],
                                        'training_time': port_experiment[regression_method]['training_time']}])],
                        ignore_index=True)

    results_path = destination or results_path
    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/port_table_v_entries.csv')
    keys = ['trace_family', 'y_metric',
            'regression_method', 'load_pattern', 'exp_type']
    results = results.groupby(keys).mean()
    results = pd.merge(results, original_results,
                       suffixes=['', '_original'], on=keys)
    results['delta_to_original_nmae'] = results['nmae'] - \
        results['nmae_original']
    results.to_csv(f'{results_path}/port_table_v_compared.csv')


if __name__ == '__main__':
    main(*parse_strategy_and_sparsing_factor())
