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

"DIstance between two images is feature representation"
function Omega.d(x::Img, y::Img)::Real
  xfeatures = squeezenet2(expanddims(cube(x.img)))
  yfeatures = squeezenet2(expanddims(cube(y.img)))
  eucl(xfeatures[1], yfeatures[1])
  # ds = map(eucl, xfeatures, yfeatures)
  # lens(:scores, ds)
  # mean(ds)
end
expanddims(x) = reshape(x, size(x)..., 1)