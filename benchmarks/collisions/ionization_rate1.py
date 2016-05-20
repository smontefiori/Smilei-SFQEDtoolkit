# ---------------------------------------------
# SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ---------------------------------------------

import math
L0 = 2.*math.pi # conversion from normalization length to wavelength

referenceAngularFrequency_SI = L0 * 3e8 /1.e-6

Main(
    geometry = "1d3v"

# number_of_patches: list of the number of patches in each dimension
number_of_patches = [ 4 ]

interpolation_order = 2

timestep = 0.05 * L0
sim_time  = 100 * L0


time_fields_frozen = 100000000000.

# SIMULATION BOX : for all space directions (in 2D & 3D use vector of doubles)
cell_length = [20.*L0]
sim_length  = [1600.*L0]

bc_em_type_x  = ["periodic"]


random_seed = 0

Species(
	species_type = "electron1",
	initPosition_type = "regular",
	initMomentum_type = "maxwell-juettner",
	n_part_per_cell= 100,
	mass = 1.0,
	charge = -1.0,
	charge_density = 1.,
	mean_velocity = [0.03, 0., 0.],
	temperature = [0.0000001]*3,
	time_frozen = 100000000.0,
	bc_part_type_west = "none",
	bc_part_type_east = "none",
	bc_part_type_south = "none",
	bc_part_type_north = "none",
	c_part_max = 10.
)

Species(
	species_type = "ion1",
	initPosition_type = "regular",
	initMomentum_type = "maxwell-juettner",
	n_part_per_cell= 100,
	mass = 1836.0*13.,
	charge = 3.0,
	charge_density = 1.,
	mean_velocity = [0., 0., 0.],
	temperature = [0.00000001]*3,
	time_frozen = 100000000.0,
	bc_part_type_west = "none",
	bc_part_type_east = "none",
	bc_part_type_south = "none",
	bc_part_type_north = "none",
	atomic_number = 13
)


# COLLISIONS
# species1    = list of strings, the names of the first species that collide
# species2    = list of strings, the names of the second species that collide
#               (can be the same as species1)
# coulomb_log = float, Coulomb logarithm. If negative or zero, then automatically computed.
Collisions(
	species1 = ["electron1"],
	species2 = ["ion1"],
	coulomb_log = 3,
	ionizing = True
)


# print_every (on screen text output) 
print_every = 100

# DIAGNOSTICS ON FIELDS
DiagFields(
	every = 1
)


# DIAGNOSTICS ON SCALARS
# every = integer, number of time-steps between each output
DiagScalar(
	every = 1
)


# DIAGNOSTICS ON PARTICLES - project the particles on a N-D arbitrary grid
# ------------------------------------------------------------------------
# output       = string: "density", "charge_density" or "jx_density"
#                parameter that describes what quantity is obtained 
# every        = integer > 0: number of time-steps between each output
# time_average = integer > 0: number of time-steps to average
# species      = list of strings, one or several species whose data will be used
# axes         = list of axes
# Each axis is a list: [_type_,_min_,_max_,_nsteps_,"logscale","edge_inclusive"]
#   _type_ is a string, one of the following options:
#      x, y, z, px, py, pz, p, gamma, ekin, vx, vy, vz, v or charge
#   The data is discretized for _type_ between _min_ and _max_, in _nsteps_ bins
#   The optional "logscale" sets the scale to logarithmic
#   The optional "edge_inclusive" forces the particles that are outside (_min_,_max_)
#     to be counted in the extrema bins
#   Example : axes = ("x", 0, 1, 30)
#   Example : axes = ("px", -1, 1, 100, "edge_inclusive")

DiagParticles(
	output = "charge_density",
	every = 20,
	species = ["ion1"],
	axes = [
		 ["x",    0,    sim_length[0],   1]
	]
)

DiagParticles(
	output = "density",
	every = 20,
	species = ["ion1"],
	axes = [
		 ["x",    0,    sim_length[0],   1]
	]
)


DiagParticles(
	output = "density",
	every = 20,
	species = ["electron1"],
	axes = [
		 ["px",    -0.05,    0.05,   1000]
	]
)

