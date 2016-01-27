using LIBLINEAR

# computation validation
iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
labels = iris[:, 5]
instances = convert(Matrix{Float64}, iris[:, 1:4]')
model = linear_train(labels[1:2:end], instances[:, 1:2:end]; verbose=true, solver_type=Cint(0))
gc()
(class, decvalues) = linear_predict(model, instances[:, 2:2:end], verbose=true)
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@assert (class .== labels[2:2:end]) == correct
println("pass 1.")

model = linear_train(labels[1:2:end], sparse(instances[:, 1:2:end]); verbose=true, solver_type=Cint(0))
gc()
(class, decvalues) = linear_predict(model, sparse(instances[:, 2:2:end]), verbose=true)
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@assert (class .== labels[2:2:end]) == correct
println("pass 2.")

# # test bias
# model = linear_train(labels[1:2:end], instances[:, 1:2:end]; verbose=true, solver_type=Cint(0))
# gc()
