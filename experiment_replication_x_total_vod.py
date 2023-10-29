import dataset_management 
import experiment
import global_variables_experiment
import warnings
from os import makedirs
import pandas as pd

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    results_path = global_variables_experiment.get_base_results_path('replication/vod')
    table_ii_columns = ['load_pattern', 'exp_type', 'regression_method', 'trace_family', 'y_metric', 'nmae', 'training_time']
    original_results = pd.DataFrame({
        'trace_family': ['VoD', 'VoD'] * 8,
        'y_metric': ['DispFrames', 'noAudioPlayed'] * 8,
        'regression_method': ['reg_tree', 'random_forest'] * 8,
        'load_pattern': (['PeriodicLoad'] * 8) + (['FlashcrowdLoad'] * 8),
        'exp_type': ['SingleApp', 'BothApps'] * 8,
        'nmae': [.12,.22,.09,.21,.13,.32,.12,.29,.10,.23,.09,.21,.10,.23,.08,.19],
        'training_time': [160,170,77000,100000,2300,2300,48000,66000,250,250,64000,88000,140,170,59000,86000]
        }
    )

    results_entry_columns = ['random_state'] + table_ii_columns
    results = pd.DataFrame(columns=results_entry_columns)
    trace_family = 'VoD'
    for trace in global_variables_experiment.TRACES[trace_family]:
        for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
            for random_state in range(2 if global_variables_experiment.IS_LOCAL else 6):
                x,y = dataset_management.parse_traces(trace, y_metric, ['X_flow.csv', 'X_port.csv', 'X_cluster.csv'])

                vod_experiment = experiment.run_experiment(x, y, y_metric, random_state=random_state)

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
                                       'nmae': vod_experiment[regression_method]['nmae'],
                                       'training_time': vod_experiment[regression_method]['training_time']}])], 
                        ignore_index=True)

    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/vod_table_ii_entries.csv')
    keys = ['trace_family', 'y_metric', 'regression_method', 'load_pattern', 'exp_type']
    results = results.groupby(keys).mean()
    results = pd.merge(results, original_results, suffixes=['', '_original'], on=keys)
    results['delta_to_original_nmae'] = results['nmae'] - results['nmae_original']
    results.to_csv(f'{results_path}/vod_table_ii_compared.csv')

