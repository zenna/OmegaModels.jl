Zygote.@adjoint function normalize(a)
  normalize(a)
end
Zygote.@nograd CartesianIndices
Zygote.@nograd lens
Zygote.@adjoint function Vec3(a, b, c)
  fx = Vec3(a, b, c)
  function pb(Δ::Real)
    @show typeof(x)
    Vec3(Δ.x, Δ.y, Δ.z)
  end

  function pb(Δ)
    @show typeof(x)
    Vec3(Δ.x, Δ.y, Δ.z)
  end  
  # Δ -> (Δ.x, Δ.y, Δ.z)
  fx, pb
end
