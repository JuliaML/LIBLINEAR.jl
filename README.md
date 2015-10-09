# LIBLINEAR

[![Build Status](https://travis-ci.org/innerlee/LIBLINEAR.jl.svg?branch=master)](https://travis-ci.org/innerlee/LIBLINEAR.jl)

Julia bindings for [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).

```julia
using RDatasets

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBLINEAR handles multi-class data automatically using a one-against-one strategy
labels = iris[:Species]

# First dimension of input data is features; second is instances
instances = convert(Array,iris[:, 1:4])'

# Train SVM on half of the data using default parameters. See the linear_train
# function in LIBLINEAR.jl for optional parameter settings.
model = linear_train(labels[1:2:end], instances[:, 1:2:end]);

# Test model on the other half of the data.
(predicted_labels, decision_values) = linear_predict(model, instances[:, 2:2:end]);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean((predicted_labels .== labels[2:2:end]))*100

```
## Notes

This package uses of the [LIBSVM](https://github.com/simonster/LIBSVM.jl) Julia package by Simon Kornblith as a reference when coding.
