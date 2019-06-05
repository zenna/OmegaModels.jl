module BlackScholes

using Omega
using Omega: invgammarv
using Omega.Prim: samplemeanᵣ
using Plots
using Lens
using Statistics

export simulate, simrv, diff_σ

# normals = normal(0, sqrt(1/nsteps) * σ)

function simulate(ω, σ; nsteps = 20)
  x = 0.0
  xs = Any[x]
  for i = 1:nsteps
    x += normal(ω, 0, sqrt(1/nsteps) * σ)
    push!(xs, x)
  end
  xs  
end

σ = invgammarv(1, 1)
# σ = uniform(0.0, 5.0)
# σ = constant(3.0)
k = 2.0
simrv = ciid(simulate, σ)
lastsim = lift(last)(simrv)

# Single obseration
diff = lift(max)(lastsim - k, 0)
diff_σ =  rid(diff, σ)
diffexp = samplemeanᵣ(diff_σ, 1000)

run() = @leval Loop => default_cbs(1000) rand(σ, diffexp ==ₛ 0.4477, 1000; alg = HMCFAST)

# Multiple observations
function diffmulti_(ω, ks)
  ls = lastsim(ω)
  [max(ls - k, 0) for k in ks]
end

nobs = 3
diffmulti = ciid(diffmulti_, [1.0, 2.0, 3.0])
diffmulti_σ = rid(diffmulti, σ)
diffmultiexp =  samplemeanᵣ(diffmulti_σ, 1000)
diffmultiexpnoise = diffmultiexp + normal(0, 0.01, (nobs,))

# Create fake data
"Generate fake data where `σ` is `σc`"
# function genfakedata(; σc = 3.0)
#   fakedatasamples = rand(diffmultiexp, 500)
#   @show fakedata = mean(fakedatasamples)
# end

runmulti() = @leval SSMHLoop => default_cbs(1000) rand(σ, diffmultiexpnoise ==ₛ [0.75901, 0.450418, 0.247978], 1000; alg = Replica)

end # module