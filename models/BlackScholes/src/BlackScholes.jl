module BlackScholes

using Omega
using Omega: invgammarv
using Omega.Prim: samplemeanᵣ
using Plots
using Lens

export simulate, simrv, diff_σ


# normals = normal(0, sqrt(1/nsteps) * σ)

function simulate(ω, σ; nsteps = 20)
  x = 0.0
  xs = [x]
  for i = 1:nsteps
    x += normal(ω, 0, sqrt(1/nsteps) * σ)
    push!(xs, x)
  end
  xs
end

σ = invgammarv(1, 1)
k = 2.0
simrv = ciid(simulate, σ)
lastsim = lift(last)(simrv)
diff = lift(max)(BlackScholes.lastsim - k, 0)
diff_σ =  rid(diff, σ)
diffexp = samplemeanᵣ(diff_σ, 1000)

run() = @leval SSMHLoop => default_cbs(1000) rand(σ, diffexp ==ₛ 0.4477, 1000; alg = SSMH)

end # module
