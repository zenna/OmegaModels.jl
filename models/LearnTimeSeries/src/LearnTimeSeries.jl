"Learns a Time Series from Data"
module LearnTimeSeries
using DiffEqFlux, Flux, DifferentialEquations, Plots
unicodeplots()
# using Omega

# TODO
# 1. Get causal data set
# 2. Separate the learning, how?
# -- (a) learn two different odes
# -- (b) structure ODE so that causal structure is there
# 
# 
# In that example, we learned the parametrs of the ode
# It seems like want is a flexible ODE family.
# OK I THINK I GEt it, the model you pass into neural ode is literally the function applied to the data
# So we just need to add some structure to that to make it causal.

# Ok where does the uncertainty come from?
# What about multiple datasets, the idea there is that
# You think alot of the parameters will be shared between people
# so its like a = p1(f(x)), b = p2(f(x))
# Ok, I think I got this!

## Setup ODE to optimize
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)

# Verify ODE solution
sol = solve(prob,Tsit5())
plot(sol)

# Generate data from the ODE
sol = solve(prob,Tsit5(),saveat=0.1)
A = sol[1,:] # length 101 vector
t = 0:0.1:10.0
scatter!(t,A)

# Build a neural network that sets the cost as the difference from the
# generated data and 1

p = param([2.2, 1.0, 2.0, 0.4]) # Initial Parameter Vector
function predict_rd() # Our 1-layer neural network
  diffeq_rd(p,prob,Tsit5(),saveat=0.1)[1,:]
end
loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function

# Optimize the parameters so the ODE's solution stays near 1

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end
# Display the ODE with the initial parameter values.
cb()
Flux.train!(loss_rd, [p], data, opt, cb = cb)

end # module
