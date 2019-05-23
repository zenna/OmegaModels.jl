module RNN

using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
# using StatsBase: wsample
# using Base.Iterators: partition

using ..NeuralODE: getdata
using ..Train: train!
N = 2
m = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  elu)

# m = gpu(m)

# function data(n = 30)
#   Ohio.NeuralODE.getdata()[:,1:30]
# end

# function loss(xs, ys)
#   l = sum(xs .- ys)
#   Flux.truncate!(m)
#   return l
# end

# opt = ADAM(0.01)
# tx, ty = (Xs[5], Ys[5])
# evalcb = () -> @show loss(tx, ty)

# function runmodel(nsteps, init)
#   init
#   x
# end

function trainrnn(; datasize = 10)
  data = getdata(; datasize = datasize)
  datait = Iterators.repeated((), 1000)
  function loss()    
    l = sum(data .- m(ones(size(data))))
    Flux.truncate!(m)
    return l
  end
  train!(loss, params(m), datait, Flux.ADAM(0.01))
end

# function sample(m, alphabet, len; temp = 1)
#   m = cpu(m)
#   Flux.reset!(m)
#   buf = IOBuffer()
#   c = rand(alphabet)
#   for i = 1:len
#     write(buf, c)
#     c = wsample(alphabet, m(onehot(c, alphabet)).data)
#   end
#   return String(take!(buf))
# end

# sample(m, alphabet, 1000) |> println

# evalcb = function ()
#   @show loss(Xs[5], Ys[5])
#   println(sample(deepcopy(m), alphabet, 100))
# end

end