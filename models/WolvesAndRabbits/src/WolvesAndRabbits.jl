module WolvesAndRabbits
using Omega
using Flux, DiffEqFlux, DifferentialEquations, Plots, DiffEqNoiseProcess
plotly()

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]

function lotka_volterra_noise(du,u,p,tt)
  du[1] = 0.1u[1]
  du[2] = 0.1u[2]
end
dt = 1//2^(4)

μ = 1.0
σ = 2.0
W = ciid(ω ->  WienerProcess(0.0,0.0,0.0; rng = ω))
# W = ciid(ω -> GeometricBrownianMotionProcess(μ,σ,0.0,1.0,1.0; rng = ω))
prob = ciid(ω -> SDEProblem(lotka_volterra,lotka_volterra_noise,[1.0,1.0],(0.0,10.0), p, noise = W(ω)))
sol = ciid(ω -> solve(prob(ω),EM()))
plot(rand(sol))

# Add an impulse
condition(u,t,integrator) = t==4 && u[1]/V<4
affect!(integrator) = integrator.u[2] += 5.0
cb = DiscreteCallback(condition,affect!)

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
