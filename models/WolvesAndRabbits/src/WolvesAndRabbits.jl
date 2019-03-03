module WolvesAndRabbits
using Omega
using Flux, DiffEqFlux, DifferentialEquations, Plots, DiffEqNoiseProcess
plotly()

# Plot results
function plotwr(data; kwargs)
  plot(data)
end

# Lotka Volterra represents dynamics of wolves and Rabbit Populations over time
function lotka_volterra(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial conditions
# u0 = constant([1.0, 1.0])
u0 = uniform(0, 2, (2,))

# Iterate over 10 time steps
tspan = constant((0.0, 20.0))

# Parameters of the simulation
# p = constant([1.5,1.0,3.0,1.0])
p = uniform(0.5, 4.0, (4,))

prob = ciid(ω -> ODEProblem(lotka_volterra, u0(ω), tspan(ω), p(ω)))
sol = lift(solve)(prob)

function sample()
  plot(rand(sol))
end

# Counter-factual model
impulse = uniform(tspan[1], tspan[2]/2.0)
condition = ciid(ω -> (u, t, integrator) -> t == impulse(ω))
affect!(integrator) = integrator.u[2] /= 2.0 
cb = DiscreteCallback(condition,affect!)

# Solution to differential equation with intervention
sol_int = ciid(ω -> solve(prob(ω),
                        EM(),
                        callback = DiscreteCallback(condition(ω), affect!),
                        tstops = impulse(ω)))

# Plot a solution from an intervened model 
function sampleint()
  t, sol_int_ = rand((impulse, sol_int))
  println("intervention occured at time $t")
  plot(sol_int_)
end              

# Suppose we observe that there are no rabbits
function totalrabbits_(ω; ndays = 10)
  sol_ = sol(ω)
  n = length(sol_)
  rabbits = [sol_[i][1] for i = (n - ndays):n]
  sum(rabbits)
end

totalrabbits = ciid(totalrabbits_)

# There are no rabbits if integrated mean value is 0
norabbits = totalrabbits ==ₛ 0.0

function samplecond1(;n = 100, alg = SSMH, kwargs...)
  samples = rand((impulse, norabbits, sol, sol_int), norabbits, n; alg = alg, kwargs...)
  impulse_, nor, sols, solcf = ntranspose(samples)
  println("intervention occured at time $(impulse_[end])")
  # display(plot(logerr.(nor)))
  display(plot(sols[end]))
  display(plot(solcf[end]))
end

# THINKING
# 1. We need another cause that will make the rabbit population fall
# 2. Cutting randomly doesn't make much sense
#    Makes more sense to say if we had culled population 
#    at its peak? at some maximum value
# 3. Maybe we need to simulate for longer
# 

# samplecond1() = plot(rand(sol, norabbits, 1000)[end])

# # But we know that at some time before there were rabbits and wolves
# usetobeboth = any(sol[1] .>ₛ 5.0) & any(sol[2] .>ₛ 5.0)

# samplecond2() = plot(rand(sol, norabbits, 1000)[end])

# # Counterfactual if we had made an intervention to cull the number of foxes would there still be no rabbits
# solcf = cond(solcf, norabbits)

# function lotka_volterra_noise(du,u,p,tt)
#   du[1] = 0.1u[1]
#   du[2] = 0.1u[2]
# end
# dt = 1//2^(4)

# μ = 1.0
# σ = 2.0
# W = ciid(ω ->  WienerProcess(0.0,0.0,0.0; rng = ω))
# # W = ciid(ω -> GeometricBrownianMotionProcess(μ,σ,0.0,1.0,1.0; rng = ω))
# prob = ciid(ω -> SDEProblem(lotka_volterra,lotka_volterra_noise,[1.0,1.0],(0.0,10.0), p, noise = W(ω)))
# sol = ciid(ω -> solve(prob(ω),EM()))


# # Verify ODE solution
# sol = solve(prob,Tsit5(), callback=cb, tstops = 4.0)
# plot(sol)

# # Generate data from the ODE
# data_sol = solve(prob,Tsit5(),saveat=0.1)
# A1 = data_sol[1,:] # length 101 vector
# A2 = data_sol[2,:] # length 101 vector
# t = 0:0.1:10.0
# scatter!(t,A1,color=[1],label = "rabbits")
# scatter!(t,A2,color=[2],label = "wolves")

# # Build a neural network that sets the cost as the difference from the
# # generated data and true data

# p = param([4., 1.0, 2.0, 0.4]) # Initial Parameter Vector
# function predict_rd() # Our 1-layer neural network
#   diffeq_rd(p,prob,Tsit5(),saveat=0.1)
# end
# loss_rd() = sum(abs2,predict_rd()-data_sol) # loss function

# # Optimize the parameters so the ODE's solution stays near 1

# data = Iterators.repeated((), 1000)
# opt = ADAM(0.1)
# cb = function () #callback function to observe training
#   #= display(loss_rd()) =#
#   # using `remake` to re-create our `prob` with current parameters `p`
#   scatter(t,A1,color=[1],label = "rabbit data")
#   scatter!(t,A2,color=[2],label = "wolves data")
#   display(plot!(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6),labels = ["rabbit model","wolf model"],color=[1 2]))
# end
# # Display the ODE with the initial parameter values.
# cb()
# Flux.train!(loss_rd, [p], data, opt, cb = cb)

# # Can we do an intervention on the Ode?


end # module
