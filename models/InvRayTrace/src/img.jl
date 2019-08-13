"A wrapper around an image, to define custom distance"
struct Img{T}
  img::T
end

"Put image in channel width height form"
cwh(img) = permutedims(img, (3, 1, 2))

"Convert an result of render (Array{Point3, 2}) into a Array{Float64, 3}"
function cube(ii)
  a = Array{Float64}(undef, (size(ii)...),3)
  for i = 1:size(a, 1), j = 1:size(a, 2)
    a[i,j,:] = ii[i,j]
  end
  a
end

eucl(x, y) = sqrt(sum((x - y) .^ 2))

"Prepare img as input into array"
pren_net(x::Img) = Float32.(expanddims(cube(x.img)))

"DIstance between two images is feature representation"
function Omega.d(x::Img, y::Img)::Real
  xfeatures = squeezenet(pren_net(x))
  yfeatures = squeezenet(pren_net(y))
  # eucl(xfeatures[1], yfeatures[1])
  ds = map(eucl, xfeatures, yfeatures)
  # lens(:scores, ds)
  Float64(mean(ds))
end
expanddims(x) = reshape(x, size(x)..., 1)