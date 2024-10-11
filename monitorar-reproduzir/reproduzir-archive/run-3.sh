#!/bin/bash

#VIDEO="https://archive.org/details/dr-strangelove-or-how-i-learned-to-stop-worrying-and-love-the-bomb"
#TIME_INIT="23:00"
#TIME_END="24:00"
VIDEO="https://www.youtube.com/watch?v=9-Jl0dxWQs8&pp=ygUEM2IxYg%3D%3D"
TIME_INIT="13:00"
TIME_END="14:00"

mpv $VIDEO --start=$TIME_INIT --end=$TIME_END --keep-open=no

mpv --script=fps.lua $VIDEO --start=$TIME_INIT --end=$TIME_END --keep-open=no &

sar -u -B -n DEV 1 90 -o sar_stats-3.data

sadf -d sar_stats-3.data -- -u -B -n DEV > sar_stats-3.csv

sadf -d sar_stats-3.data -- -u > sar_stats-3_cpu.csv
sadf -d sar_stats-3.data -- -B > sar_stats-3_pages.csv
sadf -d sar_stats-3.data -- -n DEV > sar_stats-3_DEV.csv
