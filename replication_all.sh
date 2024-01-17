#!/bin/bash

shopt -s extglob
for replication_experiment in experiment_replication_*!(stepwise.py)
do
  python3 $replication_experiment --sparsing-factor 1 &
done
