from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def initialize_agent(Ag, neurons, alpha = 1, timesteps = None, cue_location = 0, lights_on=True, verbose = False):
    if 'include_stabilization_period' in Ag.params: # initialize as needed
        
        print("Be certain all of your neurons are in the neurons list & in the proper order!!!")

        if timesteps is None:
            timesteps = int(1 / Ag.dt)  # Initialize for 1 second
        
        # we will need constant velocity, position and head direction
        Ag.save_velocity = [0,0]
        Ag.pos = [0.5,0.5]              # for now at least, hardcode location
        Ag.head_direction = [0, 0]      # A pseudorandom HD (somewhere towards the right)
        Ag.rotational_velocity = 1
        
        for t in range(timesteps):
            if verbose:
                print()
                print()
                print(f"ROUND {t + 1} ----------------------------------------")
            Ag.t = t * Ag.dt
            # Store the history so we have it
            if Ag.save_history: 
                Ag.history["t"].append(Ag.t)
                Ag.history["pos"].append(Ag.pos)
                Ag.history["vel"].append(Ag.save_velocity)
                Ag.history["head_direction"].append(Ag.head_direction)
                Ag.history["rot_vel"].append(Ag.rotational_velocity)

            # Update the neurons, including noise
            for neuron in neurons:
                if t < (timesteps // 4):
                #if t < 2:
                    neuron.update(noise_std = 0.5, lights_on = True, cue_location = 0, alpha = alpha)
                else:
                    neuron.update(noise_std = 0, lights_on = True, cue_location = 0, alpha = alpha)
            
            # Update the global t
            Ag.params['timestep'] += 1
            
    return


def run_agent(Ag, neurons, alpha = 1, timesteps = None, cue_location = 0, lights_on = True, verbose = False):
    if 'include_stabilization_period' in Ag.params: # initialize as needed
        
        print("Be certain all of your neurons are in the neurons list & in the proper order!!!")

        if timesteps is None:
            timesteps = int(1 / Ag.dt)  # Initialize for 1 second
        
        # we will need constant velocity, position and head direction
        Ag.save_velocity = [0,0]
        Ag.pos = [0.5,0.5]              # for now at least, hardcode location
        Ag.head_direction = [0, 0]      # A pseudorandom HD (somewhere towards the right)
        Ag.rotational_velocity = 1
        
        for t in range(timesteps):
            Ag.t = t * Ag.dt
            if Ag.save_history: # Store the history so we have it
                Ag.history["t"].append(Ag.t)
                Ag.history["pos"].append(Ag.pos)
                Ag.history["vel"].append(Ag.save_velocity)
                Ag.history["head_direction"].append(Ag.head_direction)
                Ag.history["rot_vel"].append(Ag.rotational_velocity)

            # Update the neurons, including noise
            for neuron in neurons:
                neuron.update(noise_std = 0.5, lights_on = lights_on, cue_location = cue_location, alpha = alpha, verbose = verbose)
            # Update the global t
            Ag.params['timestep'] += 1
    return



def plot_synaptic_drive(neuron_pop, dt = 30, t_adder = None, vmin = None, vmax = None, vertical_line = None):
    """ Neuron_pop is of the shape (number of populations, 1, timesteps)"""
    # Format vertical lines
    if type(vertical_line) in [int, float]:
        vertical_line = np.array([vertical_line])
    elif type(vertical_line) == list:
        vertical_line = np.array(vertical_line)

    N = neuron_pop.shape[-1]
    timesteps = neuron_pop.shape[0]
    # Create deepcopy to allow transformations to take place
    temp = deepcopy(neuron_pop)
    temp = temp.T
    temp = temp[::-1,::]  
    plt.figure(figsize=(50,5))
    plt.imshow(np.roll(temp, N//2, axis = 0), aspect=10, interpolation='none', vmax=vmax, vmin=vmin, cmap='viridis')
    if t_adder is None:
        plt.title("System Synaptic Drive")
    else:
        plt.title(f"System Synaptic Drive: {t_adder}")
    plt.ylabel("Internal Head Direction (deg)")
    plt.yticks(np.array([0, N//4, N//2, N//2 + N//4, N]) - .5 , [180, 90, 0,-90, -180])
    ticks_loc_list = np.arange(0,timesteps + 1, (1000 / 6))
    plt.xticks(ticks_loc_list, np.arange(0, len(ticks_loc_list) * (dt /6), (dt / 6),dtype=int))
    plt.xlabel("Time (sec)")
    if vertical_line is not None:
        for x in vertical_line:
            plt.axvline(x, c = 'red')
    plt.colorbar()
    plt.show()
    # Delete the temp variable to free space
    del temp