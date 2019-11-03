using CSV
datadir = joinpath(dirname(@__FILE__), "..", "data")
colnames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "country", "label"]
const adult_data = CSV.read(joinpath(datadir, "adult.data"), header = colnames)