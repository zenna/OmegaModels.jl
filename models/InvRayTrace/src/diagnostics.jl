# Diagnostics
using LinearAlgebra: norm

Δ(a::Sphere, b::Sphere) = norm(a.center - b.center) + abs(a.radius - b.radius)
Δ(a::Scene, b::Scene) = hausdorff(a.geoms, b.geoms)

"distance betwee two scenes"
function hausdorff(s1, s2, Δ = Δ)
  Δm(x, S) = minimum([Δ(x, y) for y in S])
  max(maximum([Δm(e, s2) for e in s1]), maximum([Δm(e, s1) for e in s2]))
end

"Sum of minimum distances"
function sumofmin(s1, s2, Δ = Δ)
  Δm(x, S) = minimum([Δ(x, y) for y in S])
  (sum([Δm(e, s2) for e in s1])+sum([Δm(e, s1) for e in s2]))/2
end

function plothist(truth, samples, plt = plot())
  distances = Δ.(truth, samples)
  histogram(distances)
end

addhausdorff(data, stage::Type{IterEnd}; groundtruth) =
  (hausdorff = Δ(data.sample, groundtruth),)
addhausdorff(data, stage; groundtruth) = nothing


function cbs(writer, logdir, n, img)
  # Render the observed img once!
  # @show cwh(img_obs.img))
  add_image!(writer, "observed", cwh(cube(img_obs.img)), 1)

  # Render img at each stage of markov chian
  renderedimg(data, stage) = nothing
  renderedimg(data, stage::Type{IterEnd}) = (img = img(data.ω).img,)

  # Save the image to tensorboard
  tbimg(data, stage) = nothing
  tbimg(data, stage::Type{IterEnd}) = 
    add_image!(writer, "renderedimg", cwh(cube(data.img)), data.i)

  # Store the score to tensorboard
  tbp(data, stage) = nothing
  tbp(data, stage::Type{IterEnd}) = add_scalar!(writer, "p", data.p, data.i)

  # Save the omegas
  # saveω(data, stage) = nothing
  # saveω(data, stage::Type{IterEnd}) = savejld(data.ω, joinpath(logdir, "omega"), data.i)

  ωcap, ωs = capturevals(:ω, Ω)
  pcap, ps = capturevals(:p, Any)

  # cbhausdorf = (data, stage) -> addhausdorff(data, stage; groundtruth = obs_scene())
  cb = idcb → (Omega.default_cbs_tpl(n)...,
               tbp,
               renderedimg → everyn(tbimg, 10),
              #  everyn(saveω, div(n, 30)),
               ωcap → everyn(30) → savejldcb(joinpath(logdir, "omegas"), backup = true, verbose = true),
               pcap → everyn(30) → savejldcb(joinpath(logdir, "ps"), backup = true, verbose = true),
              #  cbhausdorf → plotscalar(:hausdorff, "Hausdorff distance between scenes")
               )
end

function lenses(writer)
  isobs = false
  function tb_imgs(imgs...)
    imgtype = isobs ? "obs" : "learn"
    foreach(((i, img),) -> add_image!(writer, "$imgtype/l$i", img[:, :, 1]), enumerate(imgs))
    isobs = !isobs
  end

  i = 1
  function tbscores(scores)
    i += 1
    for (j, score) in enumerate(scores)
      add_scalar!(writer, "l_$j", score, i)
    end
  end
  lmap = (filters = tb_imgs, scores = tbscores)
end