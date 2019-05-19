using RunTools
using NeuralScene
using NeuralScene.Run: infer, allparams

# Run from cmdline with: julia -L hyper.jl -E 'hyper(; params = Params(tags = [:leak]))' -- --queue
function hyper(; params = Params(), n = 10)
  params_ = merge(allparams(), params)
  paramsamples = rand(params_, n)
  # paramsamples_ = rand(allparams(), n)
  # paramsamples = [merge(p, params) for p in paramsamples_]
  display.(paramsamples)
  control(infer, paramsamples)
end