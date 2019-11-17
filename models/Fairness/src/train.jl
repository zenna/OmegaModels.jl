using TensorBoardLogger

function buildquery(classifier, data, labels = data[:label])
  # @assert false
  function observation(ω)
    classification = classifier(ω).(eachrow(data))
    classification ==ₛ (labels  .- 1)
  end
  ~ observation
end

"Generate a LENS MAP"
function genlmap(logdir, n, fairmod)
  lg = TBLogger(logdir)

  # Save the omega elements
  capture_samples = x -> Dict(:ωsamples => x.ωsamples,)
  saveit_ = incsave(joinpath(logdir, "omegas.bson"); verbose = true,
                     save = BSON.bson) ∘ capture_samples

  # Evaluate taht shit
  testcb = let dataset = load_test_data()
    function f(data)
      classifier = fairmod.classifier(data.ω)
      perf = evaluate(dataset, classifier)
      println("Performance is $perf")
      TensorBoardLogger.log_value(lg, "performance", perf; step = i = data.i)
      TensorBoardLogger.log_value(lg, "Log likelihiood", data.p; step = i = data.i)
    end
  end
  Omega.Inference.runall([everyn(testcb, 100), everyn(saveit_, 100), default_cbs_tpl(n)...])
end

"Train module `fairmod`, which should have `fairmod.classifier`"
function train(fairmod; lmap, n = 10000, alg = SSMH, kwargs...)
  classifier =~ fairmod.classifier_
  obs = Fairness.buildquery(classifier, Fairness.load_data())
  # default_cbs(n)
  @leval SSMHLoop => lmap rand(Ω, cond(classifier, obs), n; alg = alg, kwargs...)
end

function allparams()
  φ = SuParams()
  φ.simname = "infer"
  φ.name = "fairness"
  φ.runname = ~ randrunname
  φ.logdir =~ ω -> logdir(runname = φ.runname(ω))
  φ.n = uniform([200, 1000, 10_000, 100_000])
  φ.runfile = joinpath(dirname(@__FILE__), "..", "scripts", "run.jl")
  φ.kernelσ = uniform([1, 500])
  φ.gitinfo = current_commit(@__FILE__)
  φ
end

function infer(φ)
  display(φ)
  # mod = φ.mod
  mod = DT16
  train(mod; n = φ.n, lmap = genlmap(φ.logdir, φ.n, mod))
end

# function paramsamples(nsamples = 30)
#   (rand(merge(allparams(), φ, Params(Dict(:samplen => i))))  for φ in enumparams(), i = 1:nsamples)
# end
# function testhyper()
#   p = rand(allparams())
#   mkpath(p.logdir)
#   infer(p)
# end

# Run from cmdline with: julia -L hyper.jl -E 'hyper(; params = Params(tags = [:leak]))' -- --queue
function hyper(; params = Params(), n = 10)
  params_ = merge(allparams(), params)
  paramsamples = rand(params_, n)
  display.(paramsamples)
  control(infer, paramsamples)
end
