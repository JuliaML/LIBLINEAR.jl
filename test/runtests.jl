using LIBLINEAR
using Base.Test


using RDatasets

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBLINEAR handles multi-class data automatically using a one-against-one strategy
labels = iris[:Species]

# First dimension of input data is features; second is instances
instances = convert(Array,iris[:, 1:4])'

# Train model on half of the data using default parameters. See the linear_train
# function in LIBLINEAR.jl for optional parameter settings.
model = linear_train(labels[1:2:end], instances[:, 1:2:end], verbose=true);

# Test model on the other half of the data.
(predicted_labels, decision_values) = linear_predict(model, instances[:, 2:2:end]);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean((predicted_labels .== labels[2:2:end]))*100
