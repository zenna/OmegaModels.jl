using SuParameters
using RunTools: control
using Fairness: allparams, infer

# Run from cmdline with: julia -L hyper.jl -E 'hyper(; params = Params(tags = [:leak]))' -- --queue
function hyper(; n = 10)
  # params_ = merge(allparams(), params)
  paramsamples = rand(allparams(), n)
  display.(paramsamples)
  control(infer, paramsamples)
end