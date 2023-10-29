import pandas as pd
import global_variables_experiment

def parse_traces(trace: str, y_metric: str, x_files: list):
    y = pd.read_csv(f'{global_variables_experiment.PASQUINIS_PATH}/{trace}/Y.csv', header=0, nrows=global_variables_experiment.NROWS, index_col=0, usecols=['TimeStamp', y_metric], low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    x = pd.DataFrame()

    for x_file in x_files:
        read_dataset = pd.read_csv(f'{global_variables_experiment.PASQUINIS_PATH}/{trace}/{x_file}',
                                header=0, index_col=0, low_memory=True, nrows=global_variables_experiment.NROWS).apply(pd.to_numeric, errors='coerce').fillna(0)
        if len(x.columns) != 0:
            x.merge(read_dataset, how="inner",
                on="TimeStamp", copy=False)
        else:
            x = pd.DataFrame(read_dataset)

    return (x, y)

