module LIBLINEAR

# package code goes here

export train, predict

# enums
const L2R_LR = int32(0)
const L2R_L2LOSS_SVC_DUAL = int32(1)
const L2R_L2LOSS_SVC = int32(2)
const L2R_L1LOSS_SVC_DUAL = int32(3)
const MCSVM_CS = int32(4)
const L1R_L2LOSS_SVC = int32(5)
const L1R_LR = int32(6)
const L2R_LR_DUAL = int32(7)
const L2R_L2LOSS_SVR  = int32(11)
const L2R_L2LOSS_SVR_DUAL = int32(12)
const L2R_L1LOSS_SVR_DUAL = int32(13)

verbosity = true

immutable FeatureNode
  index::Int32 # Question: Why not Int64?
  value::Float64
end

immutable Problem
  l::Int32
  n::Int32 # what's n????
  y::Ptr{Float64}
  x::Ptr{Ptr{FeatureNode}}
  bias::Float64
end

immutable Parameter
  solver_type::Int32

  eps::Float64
  C::Float64
  nr_weight::Int32
  weight_label::Ptr{Int32}
  weight::Ptr{Float64}
  p::Float64
  init_sol::Ptr{Float64}
end

# todo: LINEARModel
# 66-80

# set print function
let liblinear=C_NULL
  global get_liblinear
  function get_liblinear()
    if liblinear == C_NULL
      liblinear = dlopen(joinpath(Pkg.dir(), "LIBLINEAR", "deps", "liblinear.so.3"))
      ccall(dlsym(liblinear, :set_print_string_function), Void, (Ptr{Void},), cfunction(linear_print, Void, (Ptr{Uint8},)))
    end
    liblinear
  end
end

function linear_print(str::Ptr{Uint8})
    if verbosity::Bool
        print(bytestring(str))
    end
    nothing
end

# cache the function handle
macro cachedsym(symname)
    cached = gensym()
    quote
        let $cached = C_NULL
            global ($symname)
            ($symname)() = ($cached) == C_NULL ?
                ($cached = dlsym(get_liblinear(), $(string(symname)))) : $cached
        end
    end
end
@cachedsym train
@cachedsym predict_values
@cachedsym predict_probability
@cachedsym free_model_content

# helper indices_and_weights' helper
function grp2idx{T, S <: Real}(::Type{S}, labels::AbstractVector,
    label_dict::Dict{T, Int32}, reverse_labels::Vector{T})

    idx = Array(S, length(labels))
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
function indices_and_weights{T, U<:Real}(labels::AbstractVector{T},
            instances::AbstractMatrix{U},
            weights::Union(Dict{T, Float64}, Nothing)=nothing)

    label_dict = Dict{T, Int32}()
    reverse_labels = Array(T, 0)
    idx = grp2idx(Float64, labels, label_dict, reverse_labels)

    if length(labels) != size(instances, 2)
        error("""Size of second dimension of training instance matrix
        ($(size(instances, 2))) does not match length of labels
        ($(length(labels)))""")
    end

    # Construct Parameters
    if weights == nothing || length(weights) == 0
        weight_labels = Int32[]
        weights = Float64[]
    else
        weight_labels = grp2idx(Int32, keys(weights), label_dict,
            reverse_labels)
        weights = float64(values(weights))
    end

    (idx, reverse_labels, weights, weight_labels)
end

# helper
function instances2nodes{U<:Real}(instances::AbstractMatrix{U})
    nfeatures = size(instances, 1)
    ninstances = size(instances, 2)
    nodeptrs = Array(Ptr{FeatureNode}, ninstances)
    nodes = Array(FeatureNode, nfeatures + 1, ninstances)

    for i=1:ninstances
        k = 1
        for j=1:nfeatures
            nodes[k, i] = FeatureNode(int32(j), float64(instances[j, i]))
            k += 1
        end
        nodes[k, i] = FeatureNode(int32(-1), NaN)
        nodeptrs[i] = pointer(nodes, (i-1)*(nfeatures+1)+1)
    end

    (nodes, nodeptrs)
end

# helper
function instances2nodes{U<:Real}(instances::SparseMatrixCSC{U})
    ninstances = size(instances, 2)
    nodeptrs = Array(Ptr{FeatureNode}, ninstances)
    nodes = Array(FeatureNode, nnz(instances)+ninstances)

    j = 1
    k = 1
    for i=1:ninstances
        nodeptrs[i] = pointer(nodes, k)
        while j < instances.colptr[i+1]
            val = instances.nzval[j]
            nodes[k] = FeatureNode(int32(instances.rowval[j]), float64(val))
            k += 1
            j += 1
        end
        nodes[k] = FeatureNode(int32(-1), NaN)
        k += 1
    end

    (nodes, nodeptrs)
end

# train
function train{T, U<:Real}(
          # labels & data
          labels::AbstractVector{T},
          instances::AbstractMatrix{U};
          # parameters
          weights=::Union(Dict{T, Float64}, Nothing)=nothing,
          solver_type::Int32=L2R_LR,
          eps::Float64=0.001,
          C::Float64=1.0,
          p::Float64=0.1,
          # init_sol?
          verbose::Bool=false
          )
  global verbosity

  # get init_sol
  #init_sol=?

  # construct nr_weight, weight_label, weight
  (idx, reverse_labels, weights, weight_labels) = indices_and_weights(labels,
      instances, weights)

  param = Array(Parameter, 1)
  param[1] = Parameter(solver_type, eps, C, int32(length(weights), pointer(weight_labels), pointer(weights), p, nothing)) # init_sol???

  # construct problem
  (nodes, nodeptrs) = instances2nodes(instances)
  problem = Problem[Problem(int32(size(instances, 2)), 0, pointer(idx), pointer(nodeptrs))] # what's n???

  verbosity = verbose
  ptr = ccall(train(), Ptr{Void}, (Ptr{Problem}, Ptr{Parameter}), problem, param)

  model = Model() # params for model????
  finalizer(model, linear_free)
  model
end # module

# helper
linear_free(model::Model) = ccall(free_model_content(), Void, (Ptr{Void},), model.ptr)

# predict
function predict{T, U<:Real}(
          model::Model{T},
          instances::AbstractMatrix{U};
          probability_estimates::Bool=false)
  global verbosity
  ninstances = size(instances, 2)

  if size(instances, 1) != model.nfeatures
      error("Model has $(model.nfeatures) but $(size(instances, 1)) provided")
  end

  (nodes, nodeptrs) = instances2nodes(instances)
  class = Array(T, ninstances)
  nlabels = length(model.labels)
  decvalues = Array(Float64, nlabels, ninstances)

  verbosity = model.verbose
  fn = probability_estimates ? svm_predict_probability() :
      svm_predict_values()
  for i = 1:ninstances
      output = ccall(fn, Float64, (Ptr{Void}, Ptr{SVMNode}, Ptr{Float64}),
          model.ptr, nodeptrs[i], pointer(decvalues, nlabels*(i-1)+1))
      class[i] = model.labels[int(output)]
  end
end
