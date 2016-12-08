#!/bin/bash

if [ ! -f iris.data ]; then
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
fi

python3 convert.py

generateNetCDF -d gl_input -i input.dsstne -o gl_input.nc -f features_input -s samples_input -c -t analog
generateNetCDF -d gl_output -i output.dsstne -o gl_output.nc -f features_output -s samples_input -c -t analog

train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 16 -e 100

predict -b 1024 -d gl -i features_input -o features_output -n gl.nc -f input.dsstne -s predictions -r input.dsstne
