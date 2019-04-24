"Update Tensorboard"
function uptb(writer, name, field, verbose = true)
  function updateaccuracy(data)
    val = getfield(data, field)
    verbose && println("Saving $name to Tensorboard: $val")
    Tensorboard.add_scalar!(writer, name, val, data.i)
  end
end

## Std Inf Alg Params
"Optimization-specific parameters"
function infparams()
  φ = Params()
  φ[:infalg] = SSMH
  φ[:infalgargs] = infparams_(φ[:infalg])
  φ
end

"HMCFAST Specific Params"
function infparams_(::Omega.HMCFASTAlg)
  φ = Params()
  φ[:n] = uniform([200, 500, 1000, 10000])
  φ[:stepsize] = uniform([0.1, 0.01, 0.001])
  φ[:nsteps] =  uniform([100, 200, 500, 1000])
  φ[:takeevery] =  uniform([10])
  φ
end

function infparams_(any)
  φ = Params()
  φ
end

Omega.lift(:infparams_, 1)

