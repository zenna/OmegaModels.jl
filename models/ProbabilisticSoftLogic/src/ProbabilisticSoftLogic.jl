"Implementation of Probabilistic Soft Logic (PSL) in Omega"
module ProbabilisticSoftLogic

using Omega

struct Clause
end

struct Rule
  weight::Clauase
  head::Clause
  body::Clause
end

a::Clause → b::Clause = Rule(a, b)

# Rules
isfriend(A, B) ∧ isfriend(B, C) → isfriend(A, C)


# Facts
isfriend_ = Dict{Tuple{Symbol, Symbol}, SoftBool}(
  (:alice, :bob) => SoftBool(0.9),
  (:bob, :chad) => SoftBool(0.9))

function isfriend(ω, a, b)
  if (a, b) in keys(isfriend_)
    return isfriend_[(a, b)]
  else
    SoftBool(uniform(ω, 0, 1))
  end
end

# Query
q = isfriend(:alice, :chad)

rand(q, 1000)

end