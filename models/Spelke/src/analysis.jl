"Frame by frame differences"
function Δs(video)
  Δs = Float64[]
  for i = 1:length(video) - 1
    v1 = video[i]
    v2 = video[i + 1]
    push!(Δs,  Δ(v1, v2))
  end
  Δs
end
