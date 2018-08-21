using Omega
using Test

function testcoin()
  nflips = 10
  weight = betarv(2.0, 2.0)
  flips = ciid(ω -> [bernoulli(ω, weight(ω)) for i = 1:nflips])

  obs = [1.0 for i = 1:nflips]
  ps = rand(weight, flips == obs)

  @test mean(ps) > 0.5

  obs = [0.0 for i = 1:nflips]
  ps = rand(weight, flips == obs)

  @test mean(ps) < 0.5
end