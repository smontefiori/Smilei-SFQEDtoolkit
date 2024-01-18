# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------

import math
import numpy as np

# resolution
resx = 100.0
rest = 110.0
# plasma length
L = 2.0*math.pi

cell_l = L/resx
grid_l = 3.0*L
initial_pos = grid_l / 10
final_pos = initial_pos * 9
sim_time = 10.0 * math.pi

beta = (final_pos - initial_pos) / sim_time
mom_x = math.sqrt(1. / (1. - beta**2) - 1.)

pos = np.array([[initial_pos],[1]])
mom = np.array([[mom_x],[0.],[0.]])



Main(
    geometry = "1Dcartesian",
    interpolation_order = 2,
    
    cell_length = [cell_l],
    grid_length  = [grid_l],
    
    number_of_patches = [ 2 ],
    
    timestep = L/rest,
    simulation_time = sim_time,
    
    cluster_width = 1,
    
    EM_boundary_conditions = [ ['silver-muller'] ],
    
)

Species(
	name = "charges",
	position_initialization = pos,
	momentum_initialization = mom,
	pusher="borisby",
	mass = 1.0,
	charge = -1.0,
	boundary_conditions = [
		["stop", "stop"],
	],
)


