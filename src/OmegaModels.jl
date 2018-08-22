__precompile__(false)
module OmegaModels

using Omega
using DataFrames
using Tensorboard
using JLD2q
using FileIO

export infparams, infparams_, uptb, savejld, savejld2

include("common.jl")
end