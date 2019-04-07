module ProgramLearn
using Omega
using Distributions

"Generate a randoom argument"
function randargs(rng, sym, weight)
  args = []
  for i = 1:2
    p = rand(rng[@id][i])
    # a = uniform(rng[@id], 1:3)
    if p > weight
      push!(args, :x)
    elseif p > weight / 2.0
      push!(args, randexpr_(rng[@id][i], weight * weight))
    else
      push!(args, rand(rng[@id][i], 1:20))
    end
  end
  args
end

"Generate a random experiment"
function randexpr_(rng, weight = 0.5)
  primitives = [:+, :-, :*, :/]
  head = rand(rng[@id], primitives)
  args = randargs(rng[@id], head, weight)
  Expr(:call, head, args...)
end

function wrap_(rng, weight = 0.5)
  expr = randexpr_(rng, weight)
  :(x -> $(expr))
end 
randexpr = ciid(wrap_)
evalexpr_(rng) = eval(randexpr(rng))
evalexpr = ciid(evalexpr_)

xs = collect(0.00001:1.0:10.0)
fx(rng) = map(x -> Base.invokelatest(eval(wrap_(rng)), x), xs)
data = sin.(0.00001:1.0:10.0)

function run()
  randexpr = ciid(randexpr_)
  evalexpr_(rng) = eval(randexpr(rng))
  evalexpr = ciid(fx)
  exprs = rand(randexpr, evalexpr == sin.(xs))
end
end