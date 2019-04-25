export rmmodule, modulenotebook

"Removes first and last line of str (encoding file) to rm Module"
rmmodule(str) = join(split(str, "\n"; keepempty=true)[2:end-1], "\n")

"Preprocessor that replaces includes"
function replace_includes(str, included, path)
  for ex in included
      content = read(path*ex, String)
      str = replace(str, "include(\"$(ex)\")" => content)
  end
  return str
end

replace_includes(str, includede, path::Module) = replace_includes(str, include, dirname(pathof(mod)))

"Create notebook out of module"
modulenotebook(path1, path2 = dirname(path1)) =
  Literate.notebook(path1, path2; preprocess = rmmodule)

modulenotebook(mod::Module) = modulenotebook(pathof(mod), dirname(pathof(mod)))