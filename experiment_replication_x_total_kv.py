import dataset_management 
import experiment
import global_variables_experiment
import warnings
from os import makedirs
import pandas as pd

warnings.filterwarnings('ignore')

def main(random_state=42):
    results_path = global_variables_experiment.get_base_results_path('replication/kv')
    table_iii_columns = ['load_pattern', 'exp_type', 'regression_method', 'trace_family', 'y_metric', 'nmae', 'training_time']
    original_results = pd.DataFrame({
        'trace_family': ['KV', 'KV'] * 8,
        'y_metric': ['ReadsAvg', 'WritesAvg'] * 8,
        'regression_method': ['reg_tree', 'random_forest'] * 8,
        'load_pattern': (['PeriodicLoad'] * 8) + (['FlashcrowdLoad'] * 8),
        'exp_type': ['SingleApp', 'BothApps'] * 8,
        'nmae': [.03,.03,.02,.02,.04,.05,.04,.04,.02,.02,.02,.02,.04,.04,.03,.03],
        'training_time': [310,320,32000,32000,170,190,54000,54000,140,110,32000,34000,130,110,54000,55000]
        }
    )

    results_entry_columns = ['random_state'] + table_iii_columns
    results = pd.DataFrame(columns=results_entry_columns)
    trace_family = 'KV'
    for trace in global_variables_experiment.TRACES[trace_family]:
        for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
            x,y = dataset_management.parse_traces(trace, y_metric, ['X_flow.csv', 'X_port.csv', 'X_cluster.csv'])

            kv_experiment = experiment.run_experiment(x, y, y_metric, random_state=random_state)

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
                                    'nmae': kv_experiment[regression_method]['nmae'],
                                    'training_time': kv_experiment[regression_method]['training_time']}])], 
                    ignore_index=True)

    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/kv_table_iii_entries.csv')
    keys = ['trace_family', 'y_metric', 'regression_method', 'load_pattern', 'exp_type']
    results = results.groupby(keys).mean()
    results = pd.merge(results, original_results, suffixes=['', '_original'], on=keys)
    results['delta_to_original_nmae'] = results['nmae'] - results['nmae_original']
    results.to_csv(f'{results_path}/kv_table_iii_compared.csv')

if __name__ == '__main__':
    main()
