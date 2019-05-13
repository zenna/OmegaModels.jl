# No grads
Zygote.@nograd CartesianIndices
Zygote.@nograd lens
Zygote.@nograd gendata
Zygote.@nograd datarv
Zygote.@nograd rand

# Adjoints
Zygote.@adjoint function normalize(a)
  normalize(a)
end
Zygote.@adjoint function Vec3(a, b, c)
  fx = Vec3(a, b, c)
  function pb(Δ::Real)
    @show typeof(x)
    Vec3(Δ.x, Δ.y, Δ.z)
  end

  function pb(Δ)
    @show Δ
    @assert false
  end

  function pb(Δ::NamedTuple)
    # @show typeof(Δ)
    (Δ.data[1], Δ.data[2], Δ.data[3])
  end
  # Δ -> (Δ.x, Δ.y, Δ.z)
  fx, pb
end
