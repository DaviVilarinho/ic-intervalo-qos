import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

import numpy as np
from datetime import datetime
import warnings


from dataset_management import parse_traces
warnings.filterwarnings('ignore')

IS_LOCAL = False
RANDOM_STATE = None
EXPERIMENT = "histograms-and-cdfs_25-limited_merged"

PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'/home/dv/data/projects/ic-experiments/{EXPERIMENT}/{DATE}'

traces = {
    "VOD": [
        "VoD-BothApps-FlashcrowdLoad",
        "VoD-BothApps-PeriodicLoad",
        "VoD-SingleApp-FlashcrowdLoad",
        "VoD-SingleApp-PeriodicLoad"],
    "KV": [
        "KV-BothApps-FlashcrowdLoad",
        "KV-BothApps-PeriodicLoad",
        "KV-SingleApp-FlashcrowdLoad",
        "KV-SingleApp-PeriodicLoad"]
}

y_metrics = {
    "VOD": ['DispFrames', 'noAudioPlayed'],
    "KV": ["ReadsAvg", "WritesAvg"]
}

results_path = f'{BASE_RESULTS_PATH}'
for paths in [results_path]:
    try:
        os.makedirs(paths)
    except FileExistsError:
        pass


def filter_periodic(x, y, period: int):
    x = x[x.index % period == 0]
    y = y[y.index % period == 0]

    return x, y


PERIODS = [1,  # original
           2, 4, 8, 16, 32, 64, 128, 256]


for trace_family, traces in traces.items():
    for trace in traces:
        trace_name_decomposition = trace.split('-')
        trace_apps = trace_name_decomposition[1]
        trace_load = trace_name_decomposition[2]

        for y_metric in y_metrics[trace_family]:
            # smallest first
            # per switch

            x_trace, y_dataset = parse_traces(
                trace, y_metric, ['X_cluster.csv', 'X_flow.csv', 'X_port.csv'], nrows=None)

            plt.title('CDF')
            plt.ylabel('(%)')
            plt.xlabel(y_metric)
            plt.ylim(0, 25)

            color = 0
            inc_color = 200//len(PERIODS)

            for period in PERIODS:
                x_filtered, y_filtered = filter_periodic(
                    x_trace, y_dataset, period)

                hist, bins = np.histogram(
                    y_filtered, bins=30 if y_metric == 'DispFrames' else 20)
                hist = hist / np.sum(hist) * 100

                cdf = np.cumsum(hist)

                plt.plot(bins[:-1], cdf, color=(color/256, color/256, color/256), label=f'CDF período {period}')
                color += inc_color

            plt.legend()
            plt.savefig(f'{BASE_RESULTS_PATH}/{trace}_{y_metric}_dataset_density.png')
            print(f'{trace}_{y_metric}_dataset_density')
            plt.clf()
