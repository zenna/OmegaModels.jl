"Removes first and last line of str (encoding file) to rm Module"
rmmodule(str) = join(split(str, "\n"; keepempty=true)[2:end-1], "\n")