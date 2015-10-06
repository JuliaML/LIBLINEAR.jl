module LIBLINEAR

# package code goes here

export linear_train, linear_predict

# debug
function say(sth)
  println(sth)
end

# enums
const L2R_LR = Cint(0)
const L2R_L2LOSS_SVC_DUAL = Cint(1)
const L2R_L2LOSS_SVC = Cint(2)
const L2R_L1LOSS_SVC_DUAL = Cint(3)
const MCSVM_CS = Cint(4)
const L1R_L2LOSS_SVC = Cint(5)
const L1R_LR = Cint(6)
const L2R_LR_DUAL = Cint(7)
const L2R_L2LOSS_SVR  = Cint(11)
const L2R_L2LOSS_SVR_DUAL = Cint(12)
const L2R_L1LOSS_SVR_DUAL = Cint(13)

verbosity = true

immutable FeatureNode
  index::Cint
  value::Float64
end

immutable Problem
  l::Cint # num of instances
  n::Cint # num of features, including bias feature if bias >= 0
  y::Ptr{Float64} # target values
  x::Ptr{Ptr{FeatureNode}} # sparse rep. (array of feature_node) of one training vector
  bias::Float64 # if bias >= 0, isntance x becomes [x; bias]; if < 0, no bias term (default -1)
end

immutable Parameter
  solver_type::Cint

  eps::Float64
  C::Float64
  nr_weight::Cint
  weight_label::Ptr{Cint}
  weight::Ptr{Float64}
  p::Float64
  # Initial-solution specification supported only for solver L2R_LR and L2R_L2LOSS_SVC
  #init_sol::Union{Ptr{Float64},Ptr{Void}}
end

# model
type Model{T}
  ptr::Ptr{Void}
  param::Vector{Parameter}

  # prevent these from being garbage collected
  problem::Vector{Problem}
  nodes::Array{FeatureNode}
  nodeptr::Vector{Ptr{FeatureNode}}

  labels::Vector{T}
  weight_labels::Vector{Cint}
  weights::Vector{Float64}
  nfeatures::Int
  bias::Float64
  verbose::Bool
end

# set print function
let liblinear=C_NULL
  global get_liblinear
  function get_liblinear()
    if liblinear == C_NULL
      liblinear = Libdl.dlopen(joinpath(Pkg.dir(), "LIBLINEAR", "deps", "liblinear.so.3"))
      ccall(Libdl.dlsym(liblinear, :set_print_string_function), Void, (Ptr{Void},), cfunction(linear_print, Void, (Ptr{UInt8},)))
    end
    liblinear
  end
end

function linear_print(str::Ptr{UInt8})
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
                ($cached = Libdl.dlsym(get_liblinear(), $(string(symname)))) : $cached
        end
    end
end
@cachedsym train
@cachedsym predict_values
@cachedsym predict_probability
@cachedsym free_model_content
@cachedsym check_parameter

# helper indices_and_weights' helper
function grp2idx{T, S <: Real}(::Type{S}, labels::AbstractVector,
    label_dict::Dict{T, Cint}, reverse_labels::Vector{T})

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
            weights::Union{Dict{T, Float64}, Void}=nothing)

    label_dict = Dict{T, Cint}()
    reverse_labels = Array(T, 0)
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
        weight_labels = grp2idx(Cint, keys(weights), label_dict,
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
    nodeptrs = Array(Ptr{FeatureNode}, ninstances)
    nodes = Array(FeatureNode, nnz(instances)+ninstances)

    j = 1
    k = 1
    for i=1:ninstances
        nodeptrs[i] = pointer(nodes, k)
        while j < instances.colptr[i+1]
            val = instances.nzval[j]
            nodes[k] = FeatureNode(Cint(instances.rowval[j]), float64(val))
            k += 1
            j += 1
        end
        nodes[k] = FeatureNode(Cint(-1), NaN)
        k += 1
    end

    (nodes, nodeptrs)
end

# train
# - instances: instances are colums
function linear_train{T, U<:Real}(
          # labels & data
          labels::AbstractVector{T},
          instances::AbstractMatrix{U};
          # default parameters
          weights::Union{Dict{T, Float64}, Void}=nothing,
          solver_type::Cint=L2R_L2LOSS_SVC_DUAL,
          eps::Float64=Inf,
          C::Float64=1.0,
          p::Float64=0.1,
          # Initial-solution specification supported for solver L2R_LR and L2R_L2LOSS_SVC
          init_sol::Union{Ptr{Float64}, Ptr{Void}}=C_NULL,
          # problem parameter
          bias::Float64=-1.0,
          verbose::Bool=false
          )

  global verbosity
say("=1")
  # set eps
  eps = solver_type == L2R_LR || solver_type == L2R_L2LOSS_SVC ||
        solver_type == L1R_L2LOSS_SVC || solver_type == L1R_LR ? 0.01 :
        solver_type == L2R_L2LOSS_SVR ? 0.001 :
        solver_type == L2R_L2LOSS_SVC_DUAL || solver_type == L2R_L1LOSS_SVC_DUAL ||
        solver_type == MCSVM_CS || solver_type == L2R_LR_DUAL ||
        solver_type == L2R_L2LOSS_SVR_DUAL || solver_type == L2R_L1LOSS_SVR_DUAL ? 0.1 :0.001
say("=2")
  # construct nr_weight, weight_label, weight
  (idx, reverse_labels, weights, weight_labels) = indices_and_weights(labels,
      instances, weights)

# say("- labels:")
# #say(length(labels))
# #say(labels)
# say("- instances:")
# #say(size(instances))
# #say(instances)
# say("- idx:")
# #say(length(idx))
# #say(idx)
# say("- reverse_labels:")
# #say(reverse_labels)
# say("- weights:")
# #say(weights)
# say("- weight_labels:")
# #say(weight_labels)

  param = Array(Parameter, 1)
  param[1] = Parameter(solver_type, eps, C, Cint(length(weights)), pointer(weight_labels), pointer(weights), p)#, init_sol)
say("=3")
say(param[1])
  # construct problem
  (nodes, nodeptrs) = instances2nodes(instances)
say("- nodes:")
#say(nodes)
say("- nodeptrs:")
#say(nodeptrs)
  problem = Problem[Problem(Cint(size(instances, 2)), Cint(size(instances, 1)), pointer(idx), pointer(nodeptrs), bias)]
say("=4")
say(problem[1])
  verbosity = verbose

  chk = ccall(check_parameter(), Ptr{UInt8}, (Ptr{Problem}, Ptr{Parameter}), problem, param)

  if chk != C_NULL
    println("check parameter: $(bytestring(chk))")
  end
#return chk
say("=4.5")
  ptr = ccall(train(), Ptr{Void}, (Ptr{Problem}, Ptr{Parameter}), problem, param)
say("=5")
  model = Model(ptr, param, problem, nodes, nodeptrs, reverse_labels, weight_labels, weights, size(instances, 1), bias, verbose)
  finalizer(model, linear_free)
  model
end

# helper
linear_free(model::Model) = ccall(free_model_content(), Void, (Ptr{Void},), model.ptr)

# predict
# - instances: instances are colums
function linear_predict{T, U<:Real}(
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
  fn = probability_estimates ? predict_probability() :
      predict_values()
  for i = 1:ninstances
      output = ccall(fn, Float64, (Ptr{Void}, Ptr{FeatureNode}, Ptr{Float64}),
          model.ptr, nodeptrs[i], pointer(decvalues, nlabels*(i-1)+1))
      class[i] = model.labels[round(Int,output)]
  end

  (class, decvalues)
end

end # module
