module Climate

using Omega

# This model

using CSVFiles
using DataFrames
using Dates

DATADIR = "data"

# # Parse the data
co2data = DataFrame(load(joinpath(DATADIR, "co2.csv")))
tempdata = DataFrame(load(joinpath(DATADIR, "GlobalLandTemperaturesByCountry.csv")))
zimbabwetempdata = filter(x->x.Country=="Zimbabwe", tempdata)

# Aggegate co2 data by year
co2vsyear = by(co2data, :Year, Symbol("Carbon Dioxide (ppm)") => mean)

# aggegate temperature data by year
by(zimbabwetempdata, :dt => year, AverageTemperature => mean)

# The model

function model()
end
# Plot CO2 vs time
# nb plot(co2data[4], xlabel = "Tim e", ylabel = "Carbon Dioxide (ppm)")

# 


# Create a model
