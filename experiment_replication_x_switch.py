import dataset_management 
import experiment
import global_variables_experiment
import warnings
from os import makedirs
import pandas as pd

from parser import parse_strategy_and_sparsing_factor

warnings.filterwarnings('ignore')
switch_ports = {
  "SWC1": [0,1,2,3,4],
  "SWC2": [5,6,7,8,9],
  "SWC3": [10,11,12,13,14],
  "SWC4": [15,16,17,18,19],
  "SWA1": [20,21],
  "SWA2": [22,23],
  "SWA3": [24,25,26],
  "SWA4": [27,28],
  "SWA5": [29,30],
  "SWA6": [31,32,33],
  "SWB1": [34,35,36],
  "SWB2": [37,38],
  "SWB3": [39,40,41],
  "SWB4": [42,43]
}

switch_from_port = {f'{port}': switch for switch, ports in switch_ports.items() for port in ports}

def main(strategy, sparsing_factor, random_state=42):
    results_path = global_variables_experiment.get_base_results_path('replication/switch')
    table_vii_viii_columns = ['load_pattern', 'exp_type', 'regression_method', 'trace_family', 'y_metric', 'switch', 'nmae']

    results_entry_columns = ['random_state'] + table_vii_viii_columns
    results = pd.DataFrame(columns=results_entry_columns)
    for trace_family, traces in global_variables_experiment.TRACES.items():
        for trace in traces:
            for y_metric in global_variables_experiment.Y_METRICS[trace_family]:
                x_trace,y= dataset_management.parse_traces(trace, y_metric, ['X_port.csv'])


                per_switch_traces = {switch: pd.DataFrame() for switch in switch_ports.keys()}

                for feature in x_trace.columns:
                    port = feature.split('_')[0]
                    switch = switch_from_port[port]
                    per_switch_traces[switch] = x_trace[[feature]].copy()

                for switch in per_switch_traces.keys():
                    switch_experiment = experiment.run_experiment(per_switch_traces[switch], y, y_metric, random_state=random_state)

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
                                            'switch': switch,
                                            'nmae': switch_experiment[regression_method]['nmae'],
                                            }])],
                                            ignore_index=True)

    try:
        makedirs(results_path)
    except FileExistsError:
        pass

    results.to_csv(f'{results_path}/switch_table_vii_viii_final.csv')

if __name__ == '__main__':
    main(*parse_strategy_and_sparsing_factor())
