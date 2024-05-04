@testset "Validation" begin
    model = linear_train(labels[1:2:end], inst[:, 1:2:end]; verbose=true, solver_type=Cint(0))
    GC.gc()
    (class, decvalues) = linear_predict(model, inst[:, 2:2:end], verbose=true)
    correct = ones(Bool, length(class))
    correct[[30,42,43,65,67]] .= false
    @test (class .== labels[2:2:end]) == correct

    @testset "one-class" begin
        model = linear_train(labels[1:2:end], inst[:, 1:2:end]; verbose=true, solver_type=Cint(21))
        (class, decvalues) = linear_predict(model, inst[:, 2:2:end], verbose=true)
        @test all(in(("outlier", "inlier")), class)
    end
    @testset "Sparse matrix" begin
        model = linear_train(labels[1:2:end], sparse(inst[:, 1:2:end]); verbose=true, solver_type=Cint(0))
        GC.gc()
        (class, decvalues) = linear_predict(model, sparse(inst[:, 2:2:end]), verbose=true)
        correct = ones(Bool, length(class))
        correct[[30,42,43,65,67]] .= false
        @test (class .== labels[2:2:end]) == correct
    end

    @testset "Silent" begin

        model = linear_train(labels[1:2:end], inst[:, 1:2:end]; verbose=false, solver_type=Cint(0))
        GC.gc()
        (class, decvalues) = linear_predict(model, inst[:, 2:2:end], verbose=false)
        correct = ones(Bool, length(class))
        correct[[30,42,43,65,67]] .= false
        @test (class .== labels[2:2:end]) == correct

        @testset "Sparse matrix" begin
            model = linear_train(labels[1:2:end], sparse(inst[:, 1:2:end]); verbose=false, solver_type=Cint(0))
            GC.gc()
            (class, decvalues) = linear_predict(model, sparse(inst[:, 2:2:end]), verbose=false)
            correct = ones(Bool, length(class))
            correct[[30,42,43,65,67]] .= false
            @test (class .== labels[2:2:end]) == correct
        end
    end

    @testset "Regression" begin
        x = rand(1, 10000);
        y = vec(2 .* x)
        m = linear_train(y, x, solver_type=LIBLINEAR.L2R_L2LOSS_SVR, eps=1e-5)
        y_ = first(linear_predict(m, x))
        @test size(y, 1) == size(y_, 2) # TODO: provide better test
    end

end
