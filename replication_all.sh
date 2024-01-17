#!/bin/bash

for replication_experiment in experiment_replication_*  
do
  python3 $replication_experiment --sparsing-factor 1 &
done
