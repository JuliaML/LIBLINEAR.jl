# LIBLINEAR_weights.jl

Julia bindings for [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).
This version uses an instance that admits weights in the loss function per data sample.

```julia
using RDatasets, LIBLINEAR
using Printf, Statistics

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBLINEAR handles multi-class data automatically using a one-against-the rest strategy
labels = iris[:Species]

# First dimension of input data is features; second is instances
instances = convert(Matrix, iris[:, 1:4])'

# Train SVM on half of the data using default parameters. See the linear_train
# function in LIBLINEAR.jl for optional parameter settings.
model = linear_train(labels[1:2:end], instances[:, 1:2:end], verbose=true);

# Test model on the other half of the data.
(predicted_labels, decision_values) = linear_predict(model, instances[:, 2:2:end]);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean((predicted_labels .== labels[2:2:end]))*100

```
## Credits

Created by Zhizhong Li. Forked by Javier Zazo with modified [sources](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances).

This package is adapted from the [LIBSVM](https://github.com/simonster/LIBSVM.jl) Julia package by Simon Kornblith.
