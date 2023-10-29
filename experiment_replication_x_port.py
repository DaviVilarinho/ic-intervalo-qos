import dataset_management 
import experiment
import global_variables_experiment
import warnings
from os import makedirs
import pandas as pd

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    results_path = global_variables_experiment.get_base_results_path('replication/port')
    table_vi_columns = ['load_pattern', 'exp_type', 'regression_method', 'trace_family', 'y_metric', 'nmae', 'training_time']
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

    results_entry_columns = ['random_state'] + table_vi_columns
    results = pd.DataFrame(columns=results_entry_columns)
    for trace_family, traces in global_variables_experiment.TRACES.items():
        for trace in traces:
            for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
                for random_state in range(2 if global_variables_experiment.IS_LOCAL else 18):
                    x,y = dataset_management.parse_traces(trace, y_metric, ['X_port.csv'])

                    port_experiment = experiment.run_experiment(x, y, y_metric, random_state=random_state)

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


    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/port_table_vi_entries.csv')
    keys = ['trace_family', 'y_metric', 'regression_method', 'load_pattern', 'exp_type']
    results = results.groupby(keys).mean()
    results = pd.merge(results, original_results, suffixes=['', '_original'], on=keys)
    results['delta_to_original_nmae'] = results['nmae'] - results['nmae_original']
    results.to_csv(f'{results_path}/port_table_vi_compared.csv')



