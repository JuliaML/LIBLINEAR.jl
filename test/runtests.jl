using LIBLINEAR
using Test
using DelimitedFiles
using SparseArrays

# computation validation
iris   = readdlm(joinpath(dirname(@__FILE__), "iris.csv"), ',')
labels = convert(Vector{String},iris[:, 5])
inst   = convert(Matrix{Float64}, iris[:, 1:4]')
W = ones(length(labels))

include("validation.jl")
include("weights.jl")

