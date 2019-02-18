@testset "Weights" begin
    model = linear_train(labels[1:2:end], inst[:, 1:2:end]; verbose=true, solver_type=Cint(0))
    (class_withoutweights, decvalues) = linear_predict(model, inst[:, 2:2:end], verbose=true)
    GC.gc()

    wei = unique(labels[1:2:end])
    model = linear_train(labels[1:2:end], inst[:, 1:2:end]; weights=Dict(wei[1]=> 1., wei[2]=> 1., wei[3] => 1.), verbose=true, solver_type=Cint(0))
    GC.gc()
    (class, decvalues) = linear_predict(model, inst[:, 2:2:end], verbose=true)
    @test class == class_withoutweights

    model = linear_train(labels[1:2:end], inst[:, 1:2:end]; weights=Dict(wei[1]=> 0.5, wei[2]=> 1., wei[3] => 3.), verbose=true, solver_type=Cint(0))
    GC.gc()
    (class, decvalues) = linear_predict(model, inst[:, 2:2:end], verbose=true)
    correct = ones(Bool,length(class))
    correct[[28,30,32,39,42,43]] .= false
    @test (class .== labels[2:2:end]) == correct
end
