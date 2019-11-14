using CSV
datadir = joinpath(dirname(@__FILE__), "..", "data")
colnames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "country", "label"]

categories = Dict(
  :sex => ["Female", "Male"],
  :relationship => ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"], 
  :label => [">50K", "<=50K"]
)

rmmismsing(df) = df[DataFrames.completecases(df), :]

"Turn categories into numbers"
function clean_data(df)
  # @assert false
  for name in names(df)
    if name in keys(categories)
      df[!, name]
      ff(x) = ismissing(x) ? missing : findfirst((==)(strip(x)), categories[name])
      df[!, name] = map(ff, df[!, name])
    else
      println("Could not find $name")
    end
  end
  rmmismsing(df)
end

"Laod data from file"
function load_data(path = joinpath(datadir, "adult.data"))
  adult_data = CSV.read(path, header = colnames)
  clean_data(adult_data)
end