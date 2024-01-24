# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
#
# Synchrotron: radiation loss of electrons rotating in a constant
#              magnetic field
#
# In this tests case, an electron bunch is initialized per radiation
# loss models at the same positions. The magnetic field and the initial energy
# is computed so that the initial quantum parameter is equal to 1.
#
# Validation:
# - Monte-Carlo radiation loss
# - Species scalar diagnostics
# - External fields
# - Particle binning with the quantum parameter
# ----------------------------------------------------------------------------------------

import math
import datetime
import time

# ----------------------------------------------------------------------------------------
# Main parameters

c = 299792458. #(* m/s *)
electron_mass = 9.10938356e-31 # (*kg*)
electron_charge = 1.60217662e-19 # (*coulombs*)
coulombConst = 8.987551792313e9
HBar = 1.0545718e-34 # (*m^2kg/s*)
eps_0 = 8.85418782e-12

alpha = (coulombConst * electron_charge**2)/(HBar * c)

lambdar = 1e-6                                    # Normalization wavelength
wr = 2*math.pi*c/lambdar                          # Normalization frequency

Schwinger_E_field= (electron_mass**2*c**3)/(HBar*electron_charge)                  # Schwinger electric field
Enorm = electron_mass*wr*c/electron_charge        # Normalization electric field at lambda = 1e-6

l0 = 2.0*math.pi                                  # laser wavelength

chi = 1.                                         # Initial quantum parameter
B = 270.                                          # Magnetic field strength
gamma = chi * Schwinger_E_field/(Enorm*B)          # Initial gamma factor
Rsync = math.sqrt(gamma**2 - 1.)/B                # Synchrotron radius without radiation
v = math.sqrt(1.-1./gamma**2)                     # Initial particle velocity

Lx = 4*16.*Rsync
Ly = 4*16.*Rsync

n0 = 1e-19 #1e-19 #1e-5                                         # particle density

res = 128.                                        # nb of cells in one synchrotron radius

dx = Rsync/res                                    # space step
dy = Rsync/res                                    # space step
dt  = 1./math.sqrt(1./(dx*dx) + 1./(dy*dy))       # timestep given by the CFL

dt_factor = 0.4   #0.9                                 # factor on dt
dt *= dt_factor                                   # timestep used for the simulation
Tsim = (0.26e-15 * 0.2 * wr)#5000*dt/dt_factor                          # duration of the simulation

pusher = "boris"                                    # type of pusher
#pusher = "borisby"                                    # type of pusher
radiation_list = "sfqedtk-lcfa"    # List of radiation models for species
#radiation_list = "sfqedtk-bylcfa"    # List of radiation models for species
species_name_list = ["MC"]               # List of names for species

datetime = datetime.datetime.now()
random_seed = datetime.microsecond
global_every = 1 # int((Tsim/dt)/10)

timesteps = int(math.floor(Tsim / dt))
duration = (timesteps - 1) / wr

# ----------------------------------------------------------------------------------------
# Functions

# Density profile for inital location of the particles
def n0_(x,y):
        if ((x-0.5*Lx)**2 + (y-0.5*Ly)**2 <= 1000*dx):
                return n0
        else:
                return 0.
                
def n0_1(x,y):
        if ((0.5*Lx - 2500*dx) <= x <= (0.5*Lx + 2500*dx) and (0.5*Ly - 2500*dy) <= y <= (0.5*Ly + 2500*dy)):
                return n0
        else:
                return 0.
                
# def my_gaussian_density(x,y):
#     if((left_gaussian_start_x <= x <= left_gaussian_start_x + left_gaussian_length_x) and (left_gaussian_start_y <= y <= left_gaussian_start_y + left_gaussian_length_y)):
#         return (n0_electron/n_r) * np.exp(- (x - left_bunch_center_x)**2/(2.*left_bunch_rms_x**2) - (y - left_bunch_center_y)**2/(2.*left_bunch_rms_y**2))
#     else:
#         return 0

# ----------------------------------------------------------------------------------------
# Namelists

Main(
    geometry = "2Dcartesian",

    interpolation_order = 4,
    
    solve_poisson = False,

    cell_length = [dx,dy],
    grid_length  = [Lx,Ly],

    number_of_patches = [64,64],

    time_fields_frozen = Tsim,

    timestep = dt,
    simulation_time = Tsim,

    EM_boundary_conditions = [['silver-muller'],['silver-muller']],

    random_seed = int(round(time.time() * 1000)), #smilei_mpi_rank, #

    reference_angular_frequency_SI = wr

)

# ----------------------------------------------------------------------------------------
# Initialization of the constant external field

ExternalField(
    field = "Bz",
    profile = constant(B)
)

# ----------------------------------------------------------------------------------------

Species(
    name = "electron_" + species_name_list[0],
    position_initialization = "random",
    momentum_initialization = "cold",
    particles_per_cell = int(chi * 200),
    c_part_max = 1.0,
    mass = 1.0,
    charge = -1.0,
    charge_density = n0_1,
    mean_velocity = [v ,0.0, 0.0],
    temperature = [0.],
    pusher = pusher,
    boundary_conditions = [["periodic", "periodic"],["periodic", "periodic"]], #[["remove", "remove"],["remove", "remove"]],
    radiation_model = "sfqedtk-lcfa",
    radiation_photon_species = "synchro_photon",
    radiation_photon_gamma_threshold = 1e-4,
)

#   The mc synchrotron photon emitted will be stored here.
Species(
  name = "synchro_photon",
  position_initialization = "random",
  momentum_initialization = "cold",
  particles_per_cell = 0,
  mass = 0.0,
  charge = 0.0,
  number_density = 0.0,
  boundary_conditions = [["periodic", "periodic"],["periodic", "periodic"]], #[["remove", "remove"],["remove", "remove"]],
)
# ----------------------------------------------------------------------------------------
# Global parameters for the radiation reaction models
RadiationReaction(
    minimum_chi_continuous = 1e4,
    minimum_chi_discontinuous = 1e-5,
    #table_path = "/home/smonte/extend/tables_1024",
)

# ----------------------------------------------------------------------------------------
""""""
# Scalar diagnostics
DiagScalar(
    every = 1,
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
        every = global_every,
        time_average = 1,
        species = ["electron_" + species_name_list[0]],
        axes = [
            ["x", 0., Lx, 200],
            ["y", 0., Ly, 200],
        ]
    )


for i,radiation in enumerate(radiation_list):
    # Weight x chi spatial-distribution
    DiagParticleBinning(
        deposited_quantity = "weight_chi",
        every = global_every,
        time_average = 1,
        species = ["electron_" + species_name_list[0]],
        axes = [
            ["x", 0., Lx, 200],
            ["y", 0., Ly, 200],
        ]
    )


for i,radiation in enumerate(radiation_list):
    # Chi-distribution
    DiagParticleBinning(
        deposited_quantity = "weight",
        every = global_every,
        time_average = 1,
        species = ["electron_" + species_name_list[0]],
        axes = [
            ["chi", 1e-3, 1., 1000,"logscale"],
        ]
    )
"""

""""""
# # Energy distributions of synchrotron photons
DiagParticleBinning(
    deposited_quantity = "weight",
    every = global_every,
    species =["synchro_photon"], 
    axes = [["ekin", 0., gamma, int(gamma/9)]]
) 


"""
DiagParticleBinning(
    deposited_quantity = "weight",
    every = global_every,
    species =["electron_" + species_name_list[0]], 
    axes = [["chi", 0., 1.9, 500] ]
) 


DiagTrackParticles(
    species = "synchro_photon",
    every = global_every,
    attributes = [ 'x', 'y', 'weight', 'px', 'py', 'pz'],
)

Checkpoints(
    # restart_dir = "dump1",
    dump_step = timesteps-1,
    #dump_minutes = 240.,
    exit_after_dump = False,
    #keep_n_dumps = 2,
)
"""
