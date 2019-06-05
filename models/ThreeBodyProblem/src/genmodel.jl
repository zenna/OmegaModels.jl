using Omega

uniform3(ω) = [uniform(ω, -1, 1), uniform(ω, -1, 1), uniform(ω, -1, 1)]

function bodies(ω)
  nbodies = 3
  [Body(pos = uniform3(ω), mass = 1e8,
        velocity = uniform3(ω) * 0.1,
        name = "ball1$i") for i = 1:nbodies]
end

"Does s1 and s2"
function intersects(s1, s2)
  d1 = d(s1, s2)
  d2 = (s1.r + s2.r)
  a = d1 <ₛ d2
end

pairwise(bodies) = [intersects(a, b) for a in bodies, b in bodies if a != b]
anyintersect(bodies) = anyₛ(pairwise(bodies))
anyintersectseries(bodyseries) = anyₛ(map(anyintersect, bodyseries))

# using Omega
G = constant(6.67408e-11)
bodiesrv = ciid(bodies)
bodiesseriesrv = ciid(rng -> sim(bodiesrv(rng), G(rng), Δt = 0.1, t = 1, tmax = 200))
intersectsinseries = lift(anyintersectseries)(bodiesseriesrv)
rand(bodiesseriesrv, intersectsinseries, 100; alg = SSMH)[end]
replace(bodiesseriesrv, G => 2e-11)