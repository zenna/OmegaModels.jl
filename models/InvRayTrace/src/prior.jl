
function same(xs)
  a = [x1 ==ₛ x2 for x1 in xs, x2 in xs if x1 !== x2]
  all(a)
end
norma(x) = sqrt(sum(x .* x))

pairwisef(f, sc::Scene) = [f(obj1, obj2) for obj1 in sc.geoms[1:end-1], obj2 in sc.geoms[1:end-1] if obj1 !== obj2]

"Euclidean distance between all objects"
d(s1::MaterialGeom, s2::MaterialGeom) = norma(s1.center - s2.center)

"Distance between surfance color"
cold(s1::MaterialGeom, s2::MaterialGeom) = norma(s1.surface_color - s2.surface_color)

intersect(s1::MaterialGeom, s2::MaterialGeom) = d(s1, s2) <ₛ (s1.r + s2.r)
function nointersect(s1::MaterialGeom, s2::MaterialGeom)
  d1 = d(s1, s2)
  d2 = (s1.r + s2.r)
  a = d1 >ₛ d2
end

# function sall(xs)
#   @show typeof(xs)
#   @assert false
#   if isempty(xs)
#     return Omega.softtrue()
#   end
#   all(xs)
# end

"Do any objects in the scene intersect with any other"
intersect(sc::Scene) = any(pairwisef(intersects, sc))
nointersect(sc::Scene) = allₛ(pairwisef(nointersect, sc))
lift(:nointersect, 1)

"Are all objects isequidistant?"
isequidistant(sc::Scene) = same(pairwisef(d, sc))
lift(:isequidistant, 1)

"Convert Scene to DataFrame"
function scenetodf(scene::RayTrace.ListScene)
  alldf = DataFrame(x = Float64[], y = Float64[], z = Float64[], r = Float64[])
  for obj in scene.geoms
    x, y, z = obj.center
    df_ = DataFrame(x = [x], y = [y], z = [z], r = [obj.r])
    append!(alldf, df_)
  end
  alldf
end