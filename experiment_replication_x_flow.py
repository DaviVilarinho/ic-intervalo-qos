import dataset_management 
import experiment
import global_variables_experiment
import warnings
from os import makedirs
import pandas as pd

warnings.filterwarnings('ignore')

def main(random_state=42):
    results_path = global_variables_experiment.get_base_results_path('replication/flow')
    table_vi_columns = ['load_pattern', 'exp_type', 'regression_method', 'trace_family', 'y_metric', 'nmae', 'training_time']
    original_results = pd.DataFrame({
        'trace_family': ['VoD', 'KV'] * 8,
        'y_metric': ['DispFrames', 'ReadsAvg'] * 8,
        'regression_method': ['reg_tree', 'random_forest'] * 8,
        'load_pattern': (['PeriodicLoad'] * 8) + (['FlashcrowdLoad'] * 8),
        'exp_type': ['SingleApp', 'BothApps'] * 8,
        'nmae': [.12,.04,.11,.04,.15,.06,.14,.06,.12,.04,.11,.04,.11,.07,.10,.07],
        'training_time': [5,7,1200,5900,3,45,790,4800,22,3,1100,2400,3,5,480,9000]
        }
    )

    results_entry_columns = ['random_state'] + table_vi_columns
    results = pd.DataFrame(columns=results_entry_columns)
    for trace_family, traces in global_variables_experiment.TRACES.items():
        for trace in traces:
            for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
                x,y = dataset_management.parse_traces(trace, y_metric, ['X_flow.csv'])

                flow_experiment = experiment.run_experiment(x, y, y_metric, random_state=random_state)

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
                                        'nmae': flow_experiment[regression_method]['nmae'],
                                        'training_time': flow_experiment[regression_method]['training_time']}])], 
                        ignore_index=True)


    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/flow_table_vi_entries.csv')
    keys = ['trace_family', 'y_metric', 'regression_method', 'load_pattern', 'exp_type']
    results = results.groupby(keys).mean()
    results = pd.merge(results, original_results, suffixes=['', '_original'], on=keys)
    results['delta_to_original_nmae'] = results['nmae'] - results['nmae_original']
    results.to_csv(f'{results_path}/flow_table_vi_compared.csv')

if __name__ == '__main__':
    main()
