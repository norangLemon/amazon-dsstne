#!/bin/bash

# Fetch MNIST dataset
if [ ! -f train-images-idx3-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    gzip -d train-images-idx3-ubyte.gz
fi
if [ ! -f train-labels-idx1-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gzip -d train-labels-idx1-ubyte.gz
fi
if [ ! -f t10k-images-idx3-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    gzip -d t10k-images-idx3-ubyte.gz
fi
if [ ! -f t10k-labels-idx1-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gzip -d t10k-labels-idx1-ubyte.gz
fi


# Parse
if [ ! -f train-input.dsstne ] || [ ! -f train-output.dsstne ]; then
    python3 parse.py train train-images-idx3-ubyte train-labels-idx1-ubyte
fi
if [ ! -f test-input.dsstne ] || [ ! -f test-output.dsstne ]; then
    python3 parse.py test t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
fi

# Generate NetCDF files
generateNetCDF -d mnist_input  -i train-input.dsstne  -o train_input.nc  -f features_input  -s train_samples_index -c
generateNetCDF -d mnist_output -i train-output.dsstne -o train_output.nc -f features_output -s train_samples_index -c
generateNetCDF -d mnist_input  -i test-input.dsstne   -o test_input.nc   -f features_input  -s test_samples_index -m
generateNetCDF -d mnist_output -i test-output.dsstne  -o test_output.nc  -f features_output -s test_samples_index -m

# Train
rm -f gl.nc
train -c config.json -i train_input.nc -o train_output.nc -n gl.nc -b 256 -e 10

# Predict
predict -b 256 -d mnist -i features_input -o features_output -n gl.nc -f /dev/null -s predictions -r test-input.dsstne

