# LIBLINEAR

[![Build Status](https://travis-ci.org/innerlee/LIBLINEAR.jl.svg?branch=master)](https://travis-ci.org/innerlee/LIBLINEAR.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/x9jq6w5mji1u6eff?svg=true)](https://ci.appveyor.com/project/innerlee/liblinear-jl)

Julia bindings for [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).

```julia
using RDatasets, LIBLINEAR
using Printf, Statistics

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBLINEAR handles multi-class data automatically using a one-against-the rest strategy
labels = iris[:Species]

# First dimension of input data is features; second is instances
data = convert(Matrix, iris[:, 1:4])'

# Train SVM on half of the data using default parameters. See the linear_train
# function in LIBLINEAR.jl for optional parameter settings.
model = linear_train(labels[1:2:end], data[:, 1:2:end], verbose=true);

# Test model on the other half of the data.
(predicted_labels, decision_values) = linear_predict(model, data[:, 2:2:end]);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean((predicted_labels .== labels[2:2:end]))*100

```
## Credits

Created by Zhizhong Li.

This package is adapted from the [LIBSVM](https://github.com/simonster/LIBSVM.jl) Julia package by Simon Kornblith.
