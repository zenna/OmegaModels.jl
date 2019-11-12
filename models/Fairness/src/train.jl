function buildquery(classifier, data, labels = data[:label])
  # @assert false
  function observation(ω)
    classification = classifier(ω).(eachrow(data))
    # @show classification
    # @show labels .- 1
    classification ==ₛ (labels  .- 1)
  end
  ~ observation
end

"Trian module `fairmod`, which should have `fairmod.classifier`"
function train(fairmod; n = 100, alg = SSMH, kwargs...)
  classifier =~ fairmod.classifier_
  obs = Fairness.buildquery(classifier, Fairness.load_data())
  @leval SSMHLoop => default_cbs(100) rand(Ω, cond(classifier, obs), n; alg = alg, kwargs...)
end