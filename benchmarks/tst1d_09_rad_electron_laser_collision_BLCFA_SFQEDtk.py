# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
#
# Benchmark for the radiation in the collision of
# a GeV electron bunch with a counter-propagatin circularly polarized wave
#
# In this tests case, an electron bunch is initialized per radiation
# loss models at the same positions with an energy of 1 GeV near the right
# boundary of the box. They propagate to the left of the box where a circularly
# polarized laser plane wave is injected. This wave has an hyper-guassian
# profile of wavelength \lambda.
#
# Validation:
# - Continuous radiation reaction model: Landau-Lifshitz
#   with and without quantum corrections
# - Niel stochastic radiation reaction model
# - Monte-Carlo radiation reaction model without photon creation
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Importation

import math
import datetime

# ----------------------------------------------------------------------------------------
# User-defined parameters

c = 299792458
electron_mass = 9.10938356e-31
electron_charge = 1.60217662e-19
lambdar = 0.8e-6 / (2.0*math.pi)        # Normalization wavelength
wr = 2*math.pi*c/lambdar                # Normalization frequency
tr = 1. / wr                            # Normalization time

l0 = 2.0*math.pi                        # laser wavelength
t0 = l0                                 # optical cicle
dx = 0.105 #l0/resx                     # space step
dt  = 0.1 #0.95 * dx                    # timestep (0.95 x CFL)
Lx = 400*dx #30*l0                      # Domain length

n0 = 1e-19                              # bunch density

Tsim = 400 * dt #50.*t0                 # duration of the simulation
resx = 128.                             # nb of cells in one laser wavelength

start = 0                               # Laser start
fwhm = 5e-15 / tr                       # Gaussian time fwhm
duration = Tsim #50*t0                  # Laser duration
center = Tsim*0.5 #duration*0.5         # Laser profile center
order = 4                               # Laser order

#gaussian profile parameters
xvacuum=0.11770000000000014
xlength=2.5046
xcenter=1.37
xorder=2


gamma = 9784.7 #1000./0.511             # Electron bunch initial energy
v = math.sqrt(1 - 1./gamma**2)          # electron bunch initial velocity

pusher = "borisby"                         # type of pusher
radiation_list = ["sfqedtk-bylcfa"] # List of radiation models
species_name_list = ["MC"]            # Name of the species

datetime = datetime.datetime.now()
random_seed = datetime.microsecond

global_every = 200

# ----------------------------------------------------------------------------------------
# User-defined functions

# Density profile for inital location of the particles
def n0_(x):
        if (Lx - 10*dx < x < Lx - dx):
                return n0
        else:
                return 0.

# ----------------------------------------------------------------------------------------
# Namelists

Main(
    geometry = "1Dcartesian",

    interpolation_order = 4 ,

    cell_length = [dx],
    grid_length  = [Lx],

    number_of_patches = [16],

    timestep = dt,
    simulation_time = Tsim,

    EM_boundary_conditions = [['silver-muller']],

    reference_angular_frequency_SI = wr,

    random_seed = random_seed,

    print_every = global_every

)

# ----------------------------------------------------------------------------------------
# Laser definition
LaserPlanar1D(
    box_side         = "xmax",
    a0              = 8.,
    omega           = 1.,
    polarization_phi = 0.,
    ellipticity     = 0,
    time_envelope  = tgaussian(start=start,duration=duration,
                               fwhm=fwhm,
                               center=center,
                               order=order)
)

# ----------------------------------------------------------------------------------------
# Species

# Loop to create all the species
# One species per radiation implementations
for i,radiation in enumerate(radiation_list):

    Species(
        name = "electron_" + species_name_list[i],
        position_initialization = "regular",
        momentum_initialization = "cold",
        particles_per_cell = 100000,
        c_part_max = 1.0,
        mass = 1.0,
        charge = -1.0,
        charge_density = gaussian(n0, xvacuum=xvacuum, xlength=xlength, xcenter=xcenter, xorder=xorder, yvacuum=0., ylength=None, yfwhm=None, ycenter=None, yorder=2),
        mean_velocity = [v, 0.0, 0.0],
        temperature = [0.],
        pusher = pusher,
        radiation_model = radiation,
        boundary_conditions = [
            ["remove", "remove"],
        ],
    )

# ----------------------------------------------------------------------------------------
# Radiation parameters

RadiationReaction(
    minimum_chi_continuous = 1e4,
    minimum_chi_discontinuous = 1e-5,
    #table_path = "./"
)

# ----------------------------------------------------------------------------------------
# Scalar Diagnostics

DiagScalar(
    every = global_every
)

# ----------------------------------------------------------------------------------------
# Particle Binning

"""
# Loop to create all the species particle binning diagnostics
# One species per radiation implementations
for i,radiation in enumerate(radiation_list):

    # Weight spatial-distribution
    DiagParticleBinning(
        deposited_quantity = "weight",
        every = 500,
        time_average = 1,
        species = ["electron_" + species_name_list[i]],
        axes = [
            ["x", 0., Lx, 1000],
        ]
    )


for i,radiation in enumerate(radiation_list):
    # Weight x chi spatial-distribution
    DiagParticleBinning(
        deposited_quantity = "weight_chi",
        every = 500,
        time_average = 1,
        species = ["electron_" + species_name_list[i]],
        axes = [
            ["x", 0., Lx, 1000],
        ]
    )


for i,radiation in enumerate(radiation_list):
    # Chi-distribution
    DiagParticleBinning(
        deposited_quantity = "weight",
        every = [5000,6500,100],
        time_average = 1,
        species = ["electron_" + species_name_list[i]],
        axes = [
            ["chi", 1e-3, 1., 256,"logscale"],
        ]
    )

for i,radiation in enumerate(radiation_list):
    # Gamma-distribution
    DiagParticleBinning(
        deposited_quantity = "weight",
        every = [5000,6500,100],
        time_average = 1,
        species = ["electron_" + species_name_list[i]],
        axes = [
            ["gamma", 1., 1.1*gamma, 256,"logscale"],
        ]
    )
"""

# # Energy distributions of synchrotron photons
DiagParticleBinning(
    deposited_quantity = "weight",
    every = global_every,
    species =["synchro_photon"], 
    axes = [["ekin", 0., gamma, int(gamma/9)]]
) 
