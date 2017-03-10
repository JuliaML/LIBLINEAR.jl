module LIBLINEAR

export
    LinearModel,
    linear_train,
    linear_predict

# solver enums
const L2R_LR                =   Cint(0)
const L2R_L2LOSS_SVC_DUAL   =   Cint(1)
const L2R_L2LOSS_SVC        =   Cint(2)
const L2R_L1LOSS_SVC_DUAL   =   Cint(3)
const MCSVM_CS              =   Cint(4)
const L1R_L2LOSS_SVC        =   Cint(5)
const L1R_LR                =   Cint(6)
const L2R_LR_DUAL           =   Cint(7)
const L2R_L2LOSS_SVR        =   Cint(11)
const L2R_L2LOSS_SVR_DUAL   =   Cint(12)
const L2R_L1LOSS_SVR_DUAL   =   Cint(13)

verbosity = false

immutable FeatureNode
    index           ::  Cint
    value           ::  Float64
end

immutable Problem
    l               ::  Cint                    # num of instances
    n               ::  Cint                    # num of features, including bias feature if bias >= 0
    y               ::  Ptr{Float64}            # target values
    x               ::  Ptr{Ptr{FeatureNode}}   # sparse rep. (array of feature_node) of one training vector
    bias            ::  Float64                 # if bias >= 0, isntance x becomes [x; bias]; if < 0, no bias term (default -1)
end

immutable Parameter
    solver_type     ::  Cint
    eps             ::  Float64
    C               ::  Float64
    nr_weight       ::  Cint
    weight_label    ::  Ptr{Cint}
    weight          ::  Ptr{Float64}
    p               ::  Float64
    init_sol        ::  Ptr{Float64}            # Initial-solution specification supported only for solver L2R_LR and L2R_L2LOSS_SVC
end

immutable Model
    param           ::  Parameter
    nr_class        ::  Cint                    # number of class
    nr_feature      ::  Cint
    w               ::  Ptr{Float64}
    label           ::  Ptr{Cint}               # label of each class
    bias            ::  Float64
end

# model in julia
type LinearModel{T}
  solver_type       ::  Cint
  nr_class          ::  Int
  nr_feature        ::  Int
  w                 ::  Vector{Float64}
  _labels           ::  Vector{Cint}            # internal label names
  labels            ::  Vector{T}
  bias              ::  Float64
end

# get library
let liblinear = C_NULL
    global get_liblinear
    function get_liblinear()
        if liblinear == C_NULL
            libpath = joinpath(Pkg.dir(), "LIBLINEAR", "deps")
            libfile = is_windows() ?
                joinpath(libpath, "liblinear$(Sys.WORD_SIZE).dll") :
                joinpath(libpath, "liblinear.so.3")
            liblinear = Libdl.dlopen(libfile)
            ccall(Libdl.dlsym(liblinear, :set_print_string_function), Void,
                (Ptr{Void},),
                cfunction(linear_print, Void, (Ptr{UInt8},)))
        end
        liblinear
    end
end

# helper
function linear_print(str::Ptr{UInt8})
    if verbosity
        print(unsafe_string(str))
    end
    nothing
end

macro cachedsym(symname)
    cached = gensym()
    quote
        let $cached = C_NULL
            global ($symname)
            ($symname)() = ($cached) == C_NULL ?
                ($cached = Libdl.dlsym(get_liblinear(), $(string(symname)))) :
                    $cached
        end
    end
end
@cachedsym train
@cachedsym predict_values
@cachedsym predict_probability
@cachedsym free_model_content
@cachedsym check_parameter

# helper
function grp2idx{T, S <: Real}(::Type{S}, labels::AbstractVector,
    label_dict::Dict{T, Cint}, reverse_labels::Vector{T})

    idx = Array{S}(length(labels))
    nextkey = length(reverse_labels) + 1
    for i = 1:length(labels)
        key = labels[i]
        if (idx[i] = get(label_dict, key, nextkey)) == nextkey
            label_dict[key] = nextkey
            push!(reverse_labels, key)
            nextkey += 1
        end
    end
    idx
end

# helper
function indices_and_weights{T, U<:Real}(
            labels      ::  AbstractVector{T},
            instances   ::  AbstractMatrix{U},
            weights     ::  Union{Dict{T, Float64}, Void}=nothing
            )
    label_dict = Dict{T, Cint}()
    reverse_labels = Array{T}(0)
    idx = grp2idx(Float64, labels, label_dict, reverse_labels)

    if length(labels) != size(instances, 2)
        error("""Size of second dimension of training instance matrix
            ($(size(instances, 2))) does not match length of labels
            ($(length(labels)))""")
    end

    # Construct Parameters
    if weights == nothing || length(weights) == 0
        weight_labels = Cint[]
        weights = Float64[]
    else
        weight_labels = grp2idx(Cint, keys(weights), label_dict, reverse_labels)
        weights = Float64(values(weights))
    end

    (idx, reverse_labels, weights, weight_labels)
end

# helper
function instances2nodes{U<:Real}(instances::AbstractMatrix{U})
    nfeatures  = size(instances, 1)
    ninstances = size(instances, 2)
    nodeptrs   = Array{Ptr{FeatureNode}}(ninstances)
    nodes      = Array{FeatureNode}(nfeatures + 1, ninstances)

    for i = 1:ninstances
        k = 1
        for j = 1:nfeatures
            nodes[k, i] = FeatureNode(Cint(j), Float64(instances[j, i]))
            k += 1
        end
        nodes[k, i] = FeatureNode(Cint(-1), NaN)
        nodeptrs[i] = pointer(nodes, (i-1)*(nfeatures+1)+1)
    end

    (nodes, nodeptrs)
end

# helper
function instances2nodes{U<:Real}(instances::SparseMatrixCSC{U})
    ninstances = size(instances, 2)
    nodeptrs   = Array{Ptr{FeatureNode}}(ninstances)
    nodes      = Array{FeatureNode}(nnz(instances) + ninstances)

    j = 1
    k = 1
    for i = 1:ninstances
        nodeptrs[i] = pointer(nodes, k)
        while j < instances.colptr[i+1]
            val = instances.nzval[j]
            nodes[k] = FeatureNode(Cint(instances.rowval[j]), Float64(val))
            k += 1
            j += 1
        end
        nodes[k] = FeatureNode(Cint(-1), NaN)
        k += 1
    end

    (nodes, nodeptrs)
end

# train
function linear_train{T, U<:Real}(
            labels          ::  AbstractVector{T},
            instances       ::  AbstractMatrix{U};
            # optional parameters
            weights         ::  Union{Dict{T, Float64}, Void}=nothing,
            solver_type     ::  Cint=L2R_L2LOSS_SVC_DUAL,
            eps             ::  Real=Inf,
            C               ::  Real=1.0,
            p               ::  Real=0.1,
            init_sol        ::  Ptr{Float64}=convert(Ptr{Float64}, C_NULL), # initial solutions for solvers L2R_LR, L2R_L2LOSS_SVC
            bias            ::  Real=-1.0,
            verbose         ::  Bool=false
            )
    global verbosity
    verbosity = verbose
    eps, C, p, bias = map(Float64, (eps, C, p, bias))

    isinf(eps) && (eps = Dict(
        L2R_LR                  =>  0.01,
        L2R_L2LOSS_SVC_DUAL     =>  0.1,
        L2R_L2LOSS_SVC          =>  0.01,
        L2R_L1LOSS_SVC_DUAL     =>  0.1,
        MCSVM_CS                =>  0.1,
        L1R_L2LOSS_SVC          =>  0.01,
        L1R_LR                  =>  0.01,
        L2R_LR_DUAL             =>  0.1,
        L2R_L2LOSS_SVR          =>  0.001,
        L2R_L2LOSS_SVR_DUAL     =>  0.1,
        L2R_L1LOSS_SVR_DUAL     =>  0.1,
    )[solver_type])

    nfeatures = size(instances, 1) # instances are in columns
    # if bias >= 0, then one additional feature is added.
    bias >= 0 && (instances = [instances; fill(bias, 1, size(instances, 2))])

    (idx, reverse_labels, weights, weight_labels) =
        indices_and_weights(labels, instances, weights)

    param = Array{Parameter}(1)
    param[1] = Parameter(solver_type, eps, C, Cint(length(weights)),
        pointer(weight_labels), pointer(weights), p, init_sol)

    # construct problem
    (nodes, nodeptrs) = instances2nodes(instances)

    problem = Problem[Problem(Cint(size(instances, 2)),
        Cint(size(instances, 1)), pointer(idx), pointer(nodeptrs), bias)]

    chk = ccall(check_parameter(), Ptr{UInt8},
        (Ptr{Problem}, Ptr{Parameter}),
        problem, param)

    chk != convert(Ptr{UInt8}, C_NULL) &&
        error("Please check your parameters: $(unsafe_string(chk))")

    ptr = ccall(train(), Ptr{Model},
                (Ptr{Problem}, Ptr{Parameter}),
                problem, param)
    m = unsafe_wrap(Array, ptr, 1)[1]

    # extract w & _labels
    w_dim    = Int(m.nr_feature + (bias >= 0 ? 1 : 0))
    w_number = Int(m.nr_class == 2 && solver_type != MCSVM_CS ? 1 : m.nr_class)
    w        = copy(unsafe_wrap(Array, m.w, w_dim * w_number))
    _labels  = copy(unsafe_wrap(Array, m.label, m.nr_class))
    model    = LinearModel(solver_type, Int(m.nr_class), Int(m.nr_feature),
                    w, _labels, reverse_labels, m.bias)
    ccall(free_model_content(), Void, (Ptr{Model},), ptr)

    model
end

# predict
function linear_predict{T, U<:Real}(
            model                   ::  LinearModel{T},
            instances               ::  AbstractMatrix{U};
            probability_estimates   ::  Bool=false,
            verbose                 ::  Bool=false)
    global verbosity
    verbosity  = verbose
    ninstances = size(instances, 2) # instances are in columns

    size(instances, 1) != model.nr_feature &&
        error("""Model has $(model.nr_feature) features but
            $(size(instances, 1)) provided (instances are in columns)""")

    model.bias >= 0 &&
        (instances = [instances; fill(model.bias, 1, ninstances)])

    m = Array{Model}(1)
    m[1] = Model(Parameter(model.solver_type, .0, .0, Cint(0),
            convert(Ptr{Cint}, C_NULL), convert(Ptr{Float64}, C_NULL), .0,
            convert(Ptr{Float64}, C_NULL)),
            model.nr_class, model.nr_feature, pointer(model.w),
            pointer(model._labels), model.bias)

    (nodes, nodeptrs) = instances2nodes(instances)
    class = Array{T}(ninstances)
    w_number = Int(model.nr_class == 2 && model.solver_type != MCSVM_CS ?
        1 : model.nr_class)
    decvalues = Array{Float64}(w_number, ninstances)
    fn = probability_estimates ? predict_probability() :
        predict_values()
    for i = 1:ninstances
        output = ccall(fn, Float64, (Ptr{Void}, Ptr{FeatureNode}, Ptr{Float64}),
            pointer(m), nodeptrs[i], pointer(decvalues, w_number*(i-1)+1))
        class[i] = model.labels[round(Int,output)]
    end

    (class, decvalues)
end

end # module
