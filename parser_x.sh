#!/bin/bash

set -e

PARSED_DIR="parsed"
TRACES_DIR="../traces-netsoft-2017"
TRACE=$1
TMP_TRACE="/tmp/$TRACE"

mkdir -p $TMP_TRACE

for trace_file in $TRACES_DIR/$TRACE/X_*
do
    name=$(echo $trace_file | awk -F '/' '{printf $NF}')
    sort $trace_file --key 1 > $TMP_TRACE/$name
done

TARGET_DIR=$PARSED_DIR/$TRACE
mkdir -p $TARGET_DIR
join $TMP_TRACE/X_cluster.csv $TMP_TRACE/X_flow.csv -t ',' | join - $TMP_TRACE/X_port.csv -t ',' > $TARGET_DIR/X_parsed.csv
