#!/bin/bash

mpv https://archive.org/details/dr-strangelove-or-how-i-learned-to-stop-worrying-and-love-the-bomb --start=23:00 --end=24:00 --keep-open=no

mpv https://archive.org/details/dr-strangelove-or-how-i-learned-to-stop-worrying-and-love-the-bomb --start=23:00 --end=24:00 --keep-open=no &

sar -A -o sar_stats.data 1 90
sadf -d sar_stats.data -- -A > sar_stats.csv
