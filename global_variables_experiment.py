from os import uname
from datetime import datetime

IS_LOCAL = uname()[1] == "eclipse"
NROWS = 20 if IS_LOCAL else None
PASQUINIS_PATH = "../traces-netsoft-2017"
DATE = datetime.now().isoformat(timespec='seconds')

def get_base_results_path(experiment):
    return f'{"." if not IS_LOCAL else "/tmp"}/{experiment}/{DATE}'

TRACES = {
    "VoD": [
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

Y_METRICS = {
    "VoD": ['DispFrames', 'noAudioPlayed'],
    "KV": ["ReadsAvg", "WritesAvg"]
}
