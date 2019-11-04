using Lens
using Omega
using Climate

_Θ_cond = @leval SSMHLoop => default_cbs(100_000) rand(Ω, Climate.Θ_cond, 100_000; alg = Replica)

_Θ_cond_2 = @leval SSMHLoop => default_cbs(100_000) rand(Ω, Climate.Θ_cond_2, 100_000; alg = Replica)

_Θ_cond_3 = @leval SSMHLoop => default_cbs(100_000) rand(Ω, Climate.Θ_cond_3, 100_000; alg = Replica)

_Θ_cond_rcd = @leval SSMHLoop => default_cbs(100) rand(Ω, Climate.Θ_cond_rcd, 100; alg = Replica)
