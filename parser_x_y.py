# this script is intended to parse the dataset into two other datasets
# X, with all features and Y with 'dispframes'
import os
import csv
import pandas as pd

TRACES=[
    #"KV-BothApps-FlashcrowdLoad",
    #"KV-BothApps-PeriodicLoad",
    #"KV-SingleApp-FlashcrowdLoad",
    #"KV-SingleApp-PeriodicLoad",
    #"VoD-BothApps-FlashcrowdLoad",
    #"VoD-BothApps-PeriodicLoad",
    #"VoD-SingleApp-FlashcrowdLoad",
    "VoD-SingleApp-PeriodicLoad"]

DEFAULT_TEMP_DIR = 'parsed'
DEFAULT_TRACES_BASE_DIR = '../traces-netsoft-2017'

def parse_y(trace: str, traces_base_dir=DEFAULT_TRACES_BASE_DIR, temp_dir=DEFAULT_TEMP_DIR):
    with open(f'{temp_dir}/{trace}/Y_parsed.csv', 'w') as parsed:
        parsed.write('DispFrames\n')
        with open(f'{traces_base_dir}/{trace}/Y.csv', newline='') as csvfile:
            trace_reader = csv.DictReader(csvfile)
            for trace_row in trace_reader:
                parsed.write(f'{trace_row["DispFrames"]}\n')

def parse_x_y(trace: str, temp_dir=DEFAULT_TEMP_DIR, X=True, Y=True):
    try:
        os.mkdir(temp_dir)
        os.mkdir(f'{temp_dir}/{trace}')
    except FileExistsError:
        print("Já criado diretório...")
        
    if Y:
        parse_y(trace)

if __name__ == '__main__':
    parse_x_y(TRACES[0], Y=True, X=False)