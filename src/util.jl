export rmmodule, modulenotebook

"Removes first and last line of str (encoding file) to rm Module"
rmmodule(str) = join(split(str, "\n"; keepempty=true)[2:end-1], "\n")

"Create notebook out of module"
modulenotebook(mod::Module, path = joinpath(dirname(pathof(mod)), "..", "notebooks")) =
  Literate.notebook(pathof(mod), dirname(pathof(mod)); preprocess = rmmodule)