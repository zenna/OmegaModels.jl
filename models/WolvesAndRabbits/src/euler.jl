
using DataStructures

# Eulers Version

function inner(f, u, t, b, h, series)
  if t < b
    u_ = u + f(t + h, u) * h
    cons(u, inner(f, u_, t + h, b, h, series))
  else
    series
  end
end

"Eulers method (for demonstration)"
function euler(f, u0, a, b, h)
  inner(f, u0, a, b, h, nil())
end

# Lotka Volterra represents dynamics of wolves and Rabbit Populations over time
function lotka_volterra(t, u)
  x, y = u
  α, β, δ, γ = [1.5,1.0,3.0,1.0]
  dx = α*x - β*x*y
  dy = -δ*y + γ*x*y
  [dx, dy]
end

res = euler(lotka_volterra, [1.0, 1.0], 0.0, 10.0, 0.01)
xs = [x[1] for x in res]
ys = [x[2] for x in res]


# Real Version #