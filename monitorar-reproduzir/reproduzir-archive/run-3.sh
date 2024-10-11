#!/bin/bash

#VIDEO="https://archive.org/details/dr-strangelove-or-how-i-learned-to-stop-worrying-and-love-the-bomb"
VIDEO="https://www.youtube.com/watch?v=9-Jl0dxWQs8&pp=ygUEM2IxYg%3D%3D"

mpv $VIDEO --start=23:00 --end=24:00 --keep-open=no

mpv $VIDEO --start=23:00 --end=24:00 --keep-open=no &

sar -u -B -n DEV 1 90 -o sar_stats-3.data

sadf -d sar_stats-3.data -- -u -B -n DEV | awk -F';' '($3 == "idle" || $2 == "pgfree" || ($3 == "eth1" && $4 == "rxkB/s"))' > sar_stats-3.csv
