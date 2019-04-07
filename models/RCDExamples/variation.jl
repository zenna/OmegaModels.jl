using Omega
using Test

function (x::RandVar, y::RandVar)
  a = meanᵣ(varᵣ(x ∥ y))
  b = varᵣ(meanᵣ(x ∥ y))
  c = var(x)

  # Law of total variance
  @test c == a + b
end