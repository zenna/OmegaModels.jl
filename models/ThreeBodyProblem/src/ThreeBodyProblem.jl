module ThreeBodyProblem
using Omega
# using Parameters
using LinearAlgebra

export Body, viz, animate, bodies, sim, anyintersect, anyintersectseries

include("viz.jl")
include("genmodel.jl")

# Gravitational Constant
G = 6.67408e-11

# Celestial Body
mutable struct Body
  pos
  mass
  velocity
  momentum
  color
  name
  r
end

function Body(; pos, mass, velocity, color = [1.0, 1.0, 1.0], name = "sphere", r = 0.1)
  Body(pos, mass, velocity, mass * velocity, color, name, r)
end

"Squared Distance between `m1` and `m2`"
d2(m1, m2) = sum((m1 .- m2).^2)
d2(m1::Body, m2::Body) = d2(m1.pos, m2.pos)
d(x, y) = sqrt(d2(x, y))

normalize(x) = x / norm(x)

"Force due to gravity"
force(b1::Body, b2::Body, G) = (-G * b1.mass .* b2.mass) ./ d2(b1, b2) .* normalize(b1.pos - b2.pos)
# force(m1::Body, m2::Body) = force(m1.pos, m2.pos)

"Forces acting on `body` in `world`"
forces(body, world, G) = sum([force(body, x, G) for x in world if body != x])

momentum(body) = body.mass * body.velocity

"Simulate system using eulers method from `tmin` to `tmax`"
function sim(bodies, G = 6.67408e-11; Δt, t, tmax)
  bodiesthroughtime = []
  # forces = zeros(length(bodies))
  pnext = zeros(length(bodies))
  pnext = [body.momentum .+ Δt .* forces(body, bodies, G) for (i, body) in enumerate(bodies)]
  while t < tmax
    # Update the forces
    for (i, body) in enumerate(bodies)
      pnext[i] = body.momentum .+ Δt .* forces(body, bodies, G)
    end

    # Update the velocities: velocity = momentum / mass
    for body in bodies
      body.velocity = body.momentum / body.mass
    end
    
    # dpo/dt
    for body in bodies
      body.pos = body.pos + Δt * body.velocity
    end
    
    for (i, body) in enumerate(bodies)
      body.momentum = pnext[i]
    end
    t += Δt
    push!(bodiesthroughtime, deepcopy(bodies))
  end
  bodiesthroughtime
end

# Examples

function ok()
  ball1 = Body(pos = [0.0, 1.0, 0.0], mass = 1e5, velocity = [1.0, 0.0, 0.0], name = "ball1")
  ball2 = Body(pos = [0.0, 1.0, 1.0], mass = 1e5, velocity = [0.0, 0.0, 0.0], name = "ball2")
  ball3 = Body(pos = [-1.0, 0.0, 0.0], mass = 1e5, velocity = [0.0, 0.0, 0.0], name = "ball3")
  [ball1, ball2, ball3]
end

function smallworld(; t = 0, Δt = 0.1, tmax = 10.0)
  ball1 = Body(pos = [0.0, 1.0, 0.0], mass = 1e8, velocity = [0.2, 0.0, 0.0], name = "ball1")
  ball2 = Body(pos = [0.0, 1.0, 1.0], mass = 1e8, velocity = [0.0, 0.0, 0.0], name = "ball2")
  ball3 = Body(pos = [-1.0, 0.0, 0.0], mass = 1e8, velocity = [0.0, 0.0, 0.0], name = "ball3")
  sim([ball1, ball2, ball3]; Δt = Δt, t = t, tmax = tmax)
end

# Solar System
function solarsystem()
  earth = Body(pos = [0, 149.6e9, 0], mass = 6e24, velocity = [3e3, 0.0, 0.0])
  venus = Body(pos = [0, 1.0820948e11, 0], mass = 4.867e24, velocity = [35.02e3, 35.02e3, 35.02e3])
  Δt = 86000  # Time step
  tmax = 3600 * 24 * 365.25  # number of seconds in a year
  sim([earth, venus]; Δt = Δt, t = 0, tmax = tmax)
end

end