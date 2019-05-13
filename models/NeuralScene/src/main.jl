
# Create a random neural scene
const deepscene = DeepScene(rand(10))
const inlen = linearlength(Vector{Float64}, (Ray{Point3, Point3}, deepscene))
# const outlen = linearlength(Vector{Float64}, (Bool, Float64, Float64))
const outlen = 1
# const trackednet = Flux.Dense(inlen, outlen)
const midlen = 50
const trackednet = Flux.Chain(Flux.Dense(inlen, midlen),
                              Flux.Dense(midlen, outlen))


# Until Flux drops Tracker as its default Automatic Differentiation library,
# strip it out with this line:
const net_ = Flux.mapleaves(Flux.data, trackednet)

DSLearn.net(::Type{typeof(sceneintersect)}, ::Type{Ray}, ::Type{DeepScene}) = net_

# Show rendered example scene
x = ex_data()
const img = x.img

# Render the scene to get an (untrained) image
const neural_img = RayTrace.renderfunc(deepscene; x.render_params...)

# # Train the network
const params = [deepscene.ir]

# # Compute gradients
function f()
  g = gradient(Params(params)) do
    sum(RayTrace.renderfunc(deepscene; x.render_params...))
  end
end 
# # Vizualise the gradients

# # Now do real training

function train(; opt = ADAM(0.001),
                 niterations = 100,
                 imagesperbatch = 5,
                 datarv = gendata())
  netparams = vcat(map(x->[x.W, x.b], net_.layers)...)
  params_ = [deepscene.ir, netparams...]
  for i = 1:niterations
    grads = gradient(Params(params_)) do
      losses = 0f0  
      for j = 1:imagesperbatch
        @unpack rorig, img = rand(datarv)
        neural_img = RayTrace.renderfunc(deepscene; rorig = rorig, x.render_params...)
        loss = distance(neural_img, img)
        @show loss
        losses += loss  
        # push!(losses, loss)
      end
      lens(TrainLoop, (loss = losses, neural_img = neural_img, i = i))
      # sum(losses)
      losses
    end
    # @grab grads
    # @grab params_
    # @show length(grads)
    # @show grads[params_[1]]

    grads_ = map(x -> grads[x], params_)
    # @grab grads_

    zyg_update!(opt, (params_...,), (grads_...,))
  end
end

# using Lens, Callbacks, Flux
  # lmap = TrainLoop => runall([showprogress(10000), plotscalar() ∘ (nt -> (y = nt.i, x = nt.loss))])
  # @leval lmap train(; niterations = 10000, opt = ADAM(0.001))
