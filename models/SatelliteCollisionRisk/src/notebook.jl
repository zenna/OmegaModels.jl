### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 6b6c5926-6c27-11eb-3c7c-5bbf663db536
using Omega, SatelliteDynamics

# ╔═╡ 78746ea6-6c27-11eb-1c23-652354b2d444
# Declare simulation initial Epoch
epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0)

# ╔═╡ 9aee7bc2-6c28-11eb-3c67-e78b59971a0f
# Declare initial state in terms of osculating orbital elements
oe0  = [R_EARTH + 500e3, 0.0, 90.0, 0, 0, 0]

# ╔═╡ 9aef35dc-6c28-11eb-00e2-bfbca28266a7
# Convert osculating elements to Cartesean state
eci0 = sOSCtoCART(oe0, use_degrees=true)

# ╔═╡ 988a0d94-6c28-11eb-3bd7-552d9a9b6365
# Set the propagation end time to one orbit period after the start
T    = orbit_period(oe0[1])

# ╔═╡ aab097e0-6c28-11eb-29ab-e9e82babce72
epcf = epc0 + T

# ╔═╡ aab13a92-6c28-11eb-16b3-d1e506bc6b19
# Create an EarthInertialState orbit propagagator
orb  = EarthInertialState(epc0, eci0, dt=1.0,
            mass=1.0, n_grav=0, m_grav=0,
            drag=false, srp=false,
            moon=false, sun=false,
            relativity=false
)

# ╔═╡ aab23636-6c28-11eb-049f-fbea98abe245
# Propagate the orbit
t, epc, eci = sim!(orb, epcf)

# ╔═╡ 32789eda-6c28-11eb-2df9-59ced5019a01
# Test

# ╔═╡ Cell order:
# ╠═6b6c5926-6c27-11eb-3c7c-5bbf663db536
# ╠═78746ea6-6c27-11eb-1c23-652354b2d444
# ╠═9aee7bc2-6c28-11eb-3c67-e78b59971a0f
# ╠═9aef35dc-6c28-11eb-00e2-bfbca28266a7
# ╠═988a0d94-6c28-11eb-3bd7-552d9a9b6365
# ╠═aab097e0-6c28-11eb-29ab-e9e82babce72
# ╠═aab13a92-6c28-11eb-16b3-d1e506bc6b19
# ╠═aab23636-6c28-11eb-049f-fbea98abe245
# ╠═32789eda-6c28-11eb-2df9-59ced5019a01
