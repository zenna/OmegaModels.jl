using RunTools
using NeuralScene
using NeuralScene.Run: infer, allparams

# Run from cmdline with julia -L hyper.jl -E `hyper(; params = Params(runname = pleasework))' `
function hyper(; params = Params(), n = 10)
  paramsamples_ = rand(allparams(), n)
  paramsamples = [merge(params, p) for p in paramsamples_]
  display.(paramsamples)
  control(infer, paramsamples)
end