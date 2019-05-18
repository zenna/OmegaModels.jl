module Train
export zyg_update!
using Flux
using Zygote
using ..Distances
using ..GenData
using Lens
using Parameters
using RayTrace: sceneintersect, Ray, renderfunc

include("zygupdate.jl")

struct TrainLoop end
struct BatchLoop end

function train(; render_params,
                 deepscene,
                 net,
                 opt = ADAM(0.001),
                 niterations = 100,
                 imagesperbatch = 5,
                 datarv = gendata(; render_params = render_params),
                 normalize = identity)
  netparams = vcat(map(x->[x.W, x.b], net.layers)...)
  params_ = [deepscene.ir, netparams...]
  for i = 1:niterations
    grads = gradient(Params(params_)) do
      losses = 0f0  
      for j = 1:imagesperbatch
        @unpack rorig, img = rand(datarv)
        neural_img = renderfunc(deepscene; rorig = rorig, render_params...)
        loss = distance(neural_img, normalize(img))
        losses += loss 
        @show loss
        # push!(losses, loss)
        lens(BatchLoop, (loss = loss, img = img, neural_img = neural_img, i = i, j = j))
      end
      lens(TrainLoop, (loss = losses, i = i, deepscene = deepscene, net = net, opt = opt))
      # sum(losses)
      losses / imagesperbatch
    end

    grads_ = map(x -> grads[x], params_)
    zyg_update!(opt, (params_...,), (grads_...,))
  end
end

end