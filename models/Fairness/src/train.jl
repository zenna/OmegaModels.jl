function buildquery(classifier, data, labels = data[:label])
  # @assert false
  function observation(ω)
    classification = classifier(ω).(eachrow(data))
    classification ==ₛ (labels  .- 1)
  end
  ~ observation
end

PATH = "./"

"Generate a LENS MAP"
function genlmap()
  # Save the omega elements
  capture_samples = x -> Dict(:ωsamples => x.ωsamples,)
  savenet_ = incsave(joinpath(PATH, "omegas.bson"); verbose = true,
                     save = BSON.bson) ∘ capture_samples
  savenet = everyn(savenet_, 50)
end

"Trian module `fairmod`, which should have `fairmod.classifier`"
function train(fairmod; n = 100, alg = SSMH, kwargs...)
  classifier =~ fairmod.classifier_
  obs = Fairness.buildquery(classifier, Fairness.load_data())
  lmap = genlmap()
  # default_cbs(n)
  @leval SSMHLoop => lmap rand(Ω, cond(classifier, obs), n; alg = alg, kwargs...)
end