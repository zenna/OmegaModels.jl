"Removes first and last line of str (encoding file) to rm Module"
rmmodule(str) = join(split(str, "\n"; keepempty=true)[2:end-1], "\n")

modulenotebook(mod::Module, path = joinpath(dirname(pathof(mod)), "..", "notebooks")) =
  Literate.notebook(pathof(mod), path; preprocess = rmmodule)