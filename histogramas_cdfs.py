import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
import warnings


from dataset_management import parse_traces
warnings.filterwarnings('ignore')

IS_LOCAL = os.uname()[1].split('-').pop(0) == "ST"
RANDOM_STATE = 42 if IS_LOCAL else None
EXPERIMENT = "histograms-and-cdfs"

PASQUINIS_PATH = "./traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')
BASE_RESULTS_PATH = f'/tmp/{EXPERIMENT}/{DATE}'

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

            for period in PERIODS:
                x_filtered, y_filtered = filter_periodic(
                    x_trace, y_dataset, period)

                hist, bins = np.histogram(
                    y_filtered, bins=30 if y_metric == 'DispFrames' else 20)
                hist = hist / np.sum(hist) * 100

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]))
                plt.title('Histograma')
                plt.ylabel('Porcentagem das Amostras (%)')
                plt.xlabel(y_metric)
                plt.ylim(0, 100)
                plt.xlim(0, 30 if y_metric ==
                         'DispFrames' else None)

                cdf = np.cumsum(hist)

                plt.subplot(1, 2, 2)
                plt.plot(bins[:-1], cdf)
                plt.title('CDF')
                plt.ylabel('Porcentagem das Amostras (%)')
                plt.xlabel(y_metric)
                plt.ylim(0, 100)
                plt.xlim(0, 30 if y_metric ==
                         'DispFrames' else None)

                plt.savefig(
                    f'{BASE_RESULTS_PATH}/{period}_{trace}_{y_metric}_dataset.png')
                print(f'{period}_{trace}_{y_metric}_dataset')
