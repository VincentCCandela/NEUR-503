# Import packages
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------------------
# Define Custom Cell Classes according to Ajabi et al 2023 ---------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

class AngularVelocityCells(Neurons):
        """ 
        The AngularVelocityCells class will define a set of angular velocity neurons. 
        Each neuron will respond to a particular direction of movement (i.e. clockwise or counterclockwise), depending on 
            agent angular velocity.
        We will implement this class such that only two neurons of this time can exist.
        Neuron 0 will respond to counterclockwise motion (negative AV); Neuron 1 will respond to clockwise motion (positive AV).             
        """
        # We will set a maximum and minimum firing rate - to be changed according to recorded data
        # This is needed to prevent the model from "exploding", i.e. it is a built in smoothing function
        
        default_params = {"max_fr": 5,              # Max firing rate is for keeping things even
                          "min_fr": -5,             # Min firing rate is for keeping things even
                          "max_vel": 5,             # We need to cap this so it doesn't go crazy
                          "min_vel": -5,
                          "n": 2,                   # There will only ever be two AV neurons
                          "name": "AngularVelocityCells"} 
        

        def __init__(self, Agent, params={}):
            # Initialize the class parameters
            self.params = deepcopy(__class__.default_params) 
            self.params.update(params)
            
            # Set the parameters
            self.Agent = Agent                                     
            assert self.Agent.Environment.dimensionality == "2D", "Angular Velocity Only Exists for 2D Enviornments"
            assert self.params["n"] == 2, "Must have only 2 AngularVelocityCells"   # There are two neurons. Always.
            self.n = self.params["n"]
            # Vraibles for capping velocity and firingrate:
            self.min_vel = self.params['min_vel']
            self.max_vel = self.params['max_vel']
            self.max_fr = self.params['max_fr']
            self.min_fr = self.params['min_fr']

            self.preferred_directions = np.array([-1, 1]) # Neuron 0 will be CCW (neg), Neuron 1 will be CW (pos)

            super().__init__(Agent,self.params)


        def get_state(self, evaluate_at='agent', **kwargs) :
            # Neuron 0 will be CCW, Neuron 1 will be CW
            firingrate = np.zeros(2)                            # initialize firingrate
            angular_velocity = self.Agent.rotational_velocity   # get angular velocity from agent

            # Remove small angular velocities -- if the animal is not moving, then we set both firingrates at 1
            if abs(angular_velocity) <= 0.001 :       
                firingrate[0] = 1
                firingrate[1] = 1 

                if 'verbose' in kwargs.keys() and kwargs['verbose']:
                    print("AV too small")
                    print(f"Final CCW FR: {firingrate[0]}")
                    print(f"Final CW FR: {firingrate[1]}")
                    print(f"firingrate.T: {firingrate}")
                    print()
        
                return firingrate.T     # Immmediately return he firing rate because its just 1 and 1
            
            # Set firing rate
            if angular_velocity < 0:    # Counter clockwise rotation

                if 'verbose' in kwargs.keys() and kwargs['verbose']:
                    print(f"{angular_velocity} is negative; CCW")

                firingrate[0] = angular_velocity
                firingrate[1] = 1 / angular_velocity

            else:                       # Clockwise rotation    
                if 'verbose' in kwargs.keys() and kwargs['verbose']:
                    print(f"{angular_velocity} is positive; CW")

                firingrate[0] = 1 / angular_velocity
                firingrate[1] = angular_velocity

            # Set firingrate to both positive
            firingrate[0] = np.abs(firingrate[0])
            firingrate[1] = np.abs(firingrate[1])

            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                    print(f"Firingrate 0: {firingrate[0]}")
                    print(f"Firingrate 1: {firingrate[1]}")

            # Cap the firingrate as needed
            if firingrate[0] > self.max_fr or firingrate[0] == float('+inf'): 
                firingrate[0] = self.max_fr
                firingrate[1] = 1 / firingrate[0]

            if firingrate[0] < self.min_fr or firingrate[0] == float('-inf'): 
                firingrate[0] = self.min_fr
                firingrate[1] = 1 / firingrate[0]

            if firingrate[1] > self.max_fr or firingrate[1] == float('+inf'): 
                firingrate[1] = self.max_fr
                firingrate[0] = 1 / firingrate[1]

            if firingrate[1] < self.min_fr or firingrate[1] == float('-inf'): 
                firingrate[1] = self.min_fr
                firingrate[0] = 1 / firingrate[1]

            # Transpose for matmul purposes in get_state()
            firingrate = firingrate.T   

            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Final CCW FR: {firingrate[0]}")
                print(f"Final CW FR: {firingrate[1]}")
                print(f"firingrate.T: {firingrate}")
                print()

            return firingrate 

        def plot_rate_map(self, chosen_neurons="all", method="history", spikes=False, fig=None, \
                          ax=None, shape=None, colorbar=True, t_start=0, t_end=None, autosave=None, \
                            **kwargs):
            """ Custom rate map plotter for the AngularVelocityCells. For simplicity, the header is 
             the same (with the exception of the default value for "method). 
            """
            if "firingrate_limiter" not in kwargs.keys():
                kwargs['firingrate_limiter'] = self.max_fr
            
            agent_location = np.array(self.Agent.history["pos"])        # get the agent locations
            neuron_firingrate = np.array(self.history["firingrate"])    # get the neurons firingrate
            neuron_firingrate[np.where(neuron_firingrate > kwargs['firingrate_limiter'])] = 0     
            agent_location_x = agent_location[:,0]
            agent_location_y = agent_location[:,1]
            
            if chosen_neurons == "all":
                plt.figure(figsize=(10,3))
                plt.suptitle("Angular Velocity Cells")

                t = np.array(self.Agent.Environment.boundary)
                xmax = np.max(t[:,0])   # max width of the arena (x)
                xmax += xmax / 10       # add a buffer for visualization
                ymax = np.max(t[:,1])   # max height of the arena (y)
                ymax += ymax / 10       # add a buffer for visualization

                plt.subplot(1,2,1)
                plt.title("CCW")
                plt.scatter(x = agent_location_x, y = agent_location_y, c = neuron_firingrate[:,0], cmap = "viridis", s = .5, vmax = kwargs['firingrate_limiter'])
                plt.xticks(np.arange(0,xmax,0.5))
                plt.yticks(np.arange(0,ymax,0.5))
                plt.colorbar()
                
                plt.subplot(1,2,2)
                plt.title("CW")
                plt.scatter(x = agent_location_x, y = agent_location_y, c = neuron_firingrate[:,1], cmap = "viridis", s = .5, vmax = kwargs['firingrate_limiter'])
                plt.xticks(np.arange(0,xmax,0.5))
                plt.yticks(np.arange(0,ymax,0.5))
                plt.colorbar()
                plt.show()

            else:
                print ("Functionality only exists for chosen_neurons == 'all' at this time")
        
        def build_AV_to_HR_weight_matrix(self, CW, HR, weight_value = 1):
            """ Build weight matrix from angular velocity to AVHD neurons.
            Input
                CW: If the head rotation cell recieving input is Clockwise. If FALSE, means the HR cell is preferentially counterclockwise.
            """
            # Recall, Neuron 0 will be CCW, Neuron 1 will be CW
            n = HR.n
            weighted = np.zeros(n) + weight_value
            unweighted = np.zeros(n)

            if CW:
                w = np.stack((unweighted,weighted)).T
            else:
                w = np.stack((weighted,unweighted)).T

            return w


class HeadRotationCells(Neurons):
    """ 
    The HeadRotationCells class will define a set of neurons that will integrate angular velocity and 
    head direction. This is the AngularVelocityByHeadDirection (avhd) cell described in Ajabi et al 2023.

    They MUST recieve input from AngularVelocityCells and BespokeHEadDirectionCells.
    """
    default_params = {"n": 10, 
                      "name": "HeadRotationCells", 
                      "input": {}, 
                      "gamma":-1, 
                      "tau" : 0.002} 

    def __init__(self,Agent,params={}): 
        self.params = deepcopy(__class__.default_params) # to get the default param dictionary of the current class, defined in the preamble, use __class__. Then, make sure to deepcopy it, as only making a shallow copy can have unintended consequences (i.e., any modifications to it would be propagated to ALL instances of this class!).
        self.params.update(params)
        # use the parameters to setup the object itself
        self.n = self.params["n"]
        self.name = self.params["name"]
        self.input = self.params["input"]   # By default, this is an empty dictionary
        self.gamma = self.params["gamma"]
        self.firingrate = np.zeros(self.params["n"])
        self.dt = Agent.dt
        self.tau = self.params["tau"]
        self.Agent = Agent  
        super().__init__(Agent,self.params)

    def add_input(self, input_layer, w=None):
        """ Modified add_input function.
            Adds an input layer 
            input_layer is a Neurons class. As such, it has some inherent parameters which are valuable to store.
        """
        
        if w is None:   # build weight matrix if needed
            # Random
            w = np.random.normal(loc=0, scale = 1 / np.sqrt(input_layer.n), size=(self.n, input_layer.n))
        
        # Set up layer in the layer dictionary
        self.input[input_layer.name] = {}                                       # create a layer dictionary
        self.input[input_layer.name]["layer"] = input_layer                     # store the layer
        # Other, unpacked, variables
        self.input[input_layer.name]["name"]  = input_layer.name                # store the layer name
        self.input[input_layer.name]["w"] = w                                   # store the weight matrix
        self.input[input_layer.name]["n"] = input_layer.n                       # store the number of units


    def get_state(self,evaluate_at='agent',**kwargs): 
        """ 
        Get the firing rate of the current neuron, cycling through all its input layers
        If we want to add noise, it must be passed in as an argument, which will be accessed using kwargs.
        """

        # Check if we have input layers
        if np.array(self.Agent.history['t'])[-1] <= self.Agent.params['dt'] and len(self.input.keys()) == 0:
            print("WARNING: HeadRotationCells have had no input layers initialized")

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
            print(f"Getting synaptic drive of {self.name}")

        # initialize voltage and add tonic inhibition
        v = np.zeros((self.n)) + self.gamma 

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
            print(f"Gamma value: {self.gamma}")  

        # Add all the inputs, for all input layers
        for layer_name in self.input.keys():

            layer_weight = self.input[layer_name]["w"]      # weight matrix
            timestep = self.Agent.params['timestep']        # to keep track of which was the synaptic drives at the previous step.

            if timestep != 0:
                # Get the firing rate from the previous timestep
                layer_firingrate = self.input[layer_name]["layer"].history['firingrate'][timestep - 1]
            else:
                # BANDAID FOR THE ANGULAR VELOCITY INPUT BEING ONES. TODO: Suture
                if layer_name == "AngularVelocityCells":
                    layer_firingrate = np.ones(self.input[layer_name]["n"])
                else:
                    layer_firingrate = np.zeros(self.input[layer_name]["n"])
            
            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Input from layer {layer_name} : {layer_firingrate}")
         
            v += np.matmul(layer_weight, layer_firingrate)

        # get ACTIVATED VALUE with constants as defined in Ajabi et al 2023
        i =  1.1 * np.log(1 + np.exp(v - 0.25))
        
        # get FIRINGRATE (synaptic drive)
        firingrate = self.firingrate + (self.dt/self.tau) * (-1*self.firingrate + i)

        # Round to make it inlighn with other model
        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Synaptic drive of {self.name} : {firingrate}")
        
        return firingrate
    
    def build_HR_to_HDC_weight_matrix(self, CW, weight_value = 1.6248, sigma = 5, offset = 5):
        """ 
        Builds weight matrix from AVHD to HD cells 
        Takes as input"
            sigma (std of weights), 
            k (weight constant), 
            CW (if true, indicates the AVHD neurons are signaling a right shift, otherwise a left),
            offset (the max connections)
        """
        N = self.n
        # Step 1: Make the template weight matrix - not scaled to the number of neurons in the model
        var = (sigma ** 2) / (360 ** 2) * (N ** 2)  # variance in the weights - from a custom made matrix?
        i = np.ones((N,1)) * np.arange(0,N)
        j = deepcopy(i.T)
        t1 = abs(j + N - i)
        t2 = abs(i + N - j)
        t3 = abs(j - i)
        # The weight matrix is how far each inhibitory neuron is offset from the excitatory neurons it is connnected to
        weight_matrix = np.min(np.stack((t1,t2,t3)),axis = 0)   
        if CW:
            offset = offset
        else:
            offset = -1 * offset
        weight_matrix = np.roll(weight_matrix, offset, axis = -1)
    
        w = weight_value * np.exp(-1 * (weight_matrix * weight_matrix) / var)

        return w
    

# TODO: Fix this
class InhibitoryCells(Neurons):
    """ 
    The InhibitoryCells class will define a set of neurons that will provide constant inhibition.

    They MUST recieve input from some other layer in order to fire. In our case, that layer will be 
    from BespokeHeadDirectionCells.
    
    """
    default_params = {"n": 10, 
                      "name": "InhibitoryCells", 
                      "dt": 0.001,
                      "gamma":-7.5,
                      "input": {}, 
                      "tau" : 0.002} 

    def __init__(self,Agent,params={}): #<-- do not change these
        self.params = deepcopy(__class__.default_params) # to get the default param dictionary of the current class, defined in the preamble, use __class__. Then, make sure to deepcopy it, as only making a shallow copy can have unintended consequences (i.e., any modifications to it would be propagated to ALL instances of this class!).
        self.params.update(params)
        # use the parameters to setup the object itself
        self.name = self.params["name"]
        self.input = self.params["input"]
        self.gamma = self.params["gamma"]
        self.dt = Agent.dt
        self.tau = self.params["tau"]
        self.firingrate = np.zeros(self.params["n"])
        super().__init__(Agent,self.params)

    def add_input(self, input_layer, w=None):
        """ Modified add_input function.
            Adds an input layer 
            input_layer is a Neurons class. As such, it has some inherent parameters.
        """
        if len(self.input) > 1:
            print("Only one input can be added to the inhibitory cells")
            return

        # Else, add the input layer
        if w is None:   # build weight matrix if needed
            # Random
            w = np.random.normal(loc=0, scale = 1 / np.sqrt(input_layer.n), size=(self.n, input_layer.n))
        
        # Set up the layer  
        self.input[input_layer.name] = {}                                       # Empty layer dictionary
        self.input[input_layer.name]["layer"] = input_layer                     # the layer itself
        # Other, unpacked, variables
        self.input[input_layer.name]["name"]  = input_layer.name                # layer name
        self.input[input_layer.name]["w"] = w                                   # weight matrix
        #self.input[input_layer.name]["firingrate"] = np.zeros(input_layer.n)   # firing rate of the cells
        self.input[input_layer.name]["n"] = input_layer.n                       # number of units


    def get_state(self,evaluate_at='agent',**kwargs): # <-- do not change these
        """ Get the firing rate of the current neuron, cycling through all its input layers"""
        
        # Check if there are inputs
        if np.array(self.Agent.history['t'])[-1] <= self.Agent.params['dt'] and len(self.input.keys()) == 0:
            print("WARNING: InhibitoryCells have had no input layers initialized")
        
        if 'verbose' in kwargs.keys() and kwargs['verbose']:
            print(f"Getting synaptic drive of {self.name}")

        # get VOLTAGE
        v = np.zeros((self.n)) + self.gamma 

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Gamma : {self.gamma }")

        for layer_name in self.input.keys():
            layer_weight = self.input[layer_name]["w"]                      # weight matrix
            timestep = self.Agent.params['timestep']
            if timestep != 0:
                # Get the firing rate from the previous timestep
                layer_firingrate = self.input[layer_name]["layer"].history['firingrate'][timestep - 1]
            else:
                # Firing was zero
                layer_firingrate = np.zeros(self.input[layer_name]["n"])

            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Input from layer {layer_name} : {layer_firingrate}")

            v += np.matmul(layer_weight, layer_firingrate)                  # HR voltage

        # get ACTIVATED VALUE
        i =  1.1 * np.log(1 + np.exp(v - 0.25))

        # get FIRINGRATE (synaptic drive)
        firingrate = self.firingrate + (self.dt/self.tau) * (-1*self.firingrate + i)
        
        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Synaptic drive of {self.name} : {firingrate}")
        
        return firingrate

    
    def build_I_to_HDC_weight_matrix(self, weight_value = 0.0432, sigma = 15):
        """ 
        Build weight matrix detailing connections between inhibitory neurons and head direction cells
        Takes as input the number of neurons, a standard deviation for the weights, and a weight constant
        Returns a matrix
        """
        N = self.n
        # Step 1: Make the template weight matrix - not scaled to the number of neurons in the model
        var = (sigma ** 2) / (360 ** 2) * (N ** 2)  # NOT 100% sure how this is calculated
        i = np.ones((N,1)) * np.arange(N)           
        j = deepcopy(i.T)
        t1 = abs(j + N - i)
        t2 = abs(i + N - j)
        t3 = abs(j - i)
        # The weight matrix is how far each inhibitory neuron is offset from the excitatory neurons it is connnected to
        weight_matrix = np.min(np.stack((t1,t2,t3)),axis = 0)   
        
        # Step 3: Create the finalized matrix
        w = weight_value * (np.exp(((weight_matrix * weight_matrix) * -1 ) / var) -1)

        return w


class BespokeHeadDirectionCells(Neurons):
    """ 
    The BespokeHeadDirectionCells class will define a set of neurons that will estimate head direction.

    They MUST recieve input from some layers 
    (InhibitoryNeurons, GainNeurons, VisualNeurons, AngularVelocityByHeadDirectionNeurons (HeadRotation / wumbo))

    This will be built according to Ajabi et al 2023
    """
    default_params = {"n": 10, 
                      "name": "BespokeHeadDirectionCells", 
                      "input": {}, 
                      "gamma":-1.5,
                      "dt" : 0.001, 
                      "tau" : 0.01} 

    def __init__(self,Agent,params={}): #<-- do not change these
        self.params = deepcopy(__class__.default_params) # to get the default param dictionary of the current class, defined in the preamble, use __class__. Then, make sure to deepcopy it, as only making a shallow copy can have unintended consequences (i.e., any modifications to it would be propagated to ALL instances of this class!).
        self.params.update(params)
        # use the parameters to setup the object itself
        self.name = self.params["name"]
        self.input = self.params["input"]
        self.gamma = self.params["gamma"]
        self.dt = Agent.dt
        self.tau = self.params["tau"]
        self.firingrate = np.zeros(self.params["n"])
        self.Agent = Agent

        super().__init__(Agent,self.params)

    def add_input(self, input_layer, w=None):
        """ Modified add_input function.
            Adds an input layer 
            input_layer is a Neurons class. As such, it has some inherent parameters.
        """
        
        if w is None:   # build weight matrix if needed
            # Random
            w = np.random.normal(loc=0, scale = 1 / np.sqrt(input_layer.n), size=(self.n, input_layer.n))
        
        # Set up layer in the layer dictionary
        self.input[input_layer.name] = {}                           # layer dict
        self.input[input_layer.name]["layer"] = input_layer         # store the layer itself
        # Other, unpacked, variables
        self.input[input_layer.name]["name"]  = input_layer.name    # store the layer name
        self.input[input_layer.name]["w"] = w                       # store the layer weight matrix
        self.input[input_layer.name]["firingrate"] = np.zeros(input_layer.n)    # firing rate of the cells
        self.input[input_layer.name]["n"] = input_layer.n           # store the number of units

    def get_state(self,evaluate_at='agent',**kwargs): # <-- do not change these
        """ Get the firing rate of the current neuron, cycling through all its input layers
        If we want to add noise, it must be passed in the function (i.e. part of kwargs)
        """
        
        # Check if there are inputs
        if np.array(self.Agent.history['t'])[-1] <= self.Agent.params['dt'] and len(self.input.keys()) == 0:
            print("WARNING: BespokeHeadDirectionCells have had no input layers initialized")

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Getting Synaptic drive of {self.name}")
         
        # initialize VOLTAGE
        v = np.zeros((self.n)) + self.gamma   

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Gamma: {self.gamma}")

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Voltage after the addition of gamma : {v}")
        
        # Add all the inputs, for all input layers
        for layer_name in self.input.keys():
            
            layer_weight = self.input[layer_name]["w"]
            timestep = self.Agent.params['timestep']
            if timestep != 0:
                # Get the firing rate from the previous timestepå
                layer_firingrate = self.input[layer_name]["layer"].history['firingrate'][timestep - 1]
            else:
                # Firing was zero
                layer_firingrate = np.zeros(self.input[layer_name]["n"])

            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Input from {layer_name} : {layer_firingrate}")
                print(f"Layer weight: {layer_weight}")
            v += np.matmul(layer_weight, layer_firingrate)

            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Voltage after the addition of {layer_name} : {v}")
        
        # Add noise as needed
        if "noise" in kwargs.keys() and type(kwargs['noise']) in [float, int]:
            v += kwargs['noise']

        if "noise_std" in kwargs.keys() and type(kwargs['noise_std']) in [float, int]:
            t = np.random.normal(0,kwargs['noise_std'], size = self.n) / (self.Agent.params['timestep'] + 1)
            #t = temp / (self.Agent.params['timestep'] + 1)
            v += t

            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Input from noise : {t}")
        
        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Voltage after the addition of noise : {v}")

        # get ACTIVATED VALUE
        i =  1.1 * np.log(1 + np.exp(v - 0.25))

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"activated value : {i}")

        # get FIRINGRATE (synaptic drive)
        firingrate = self.firingrate + (self.dt/self.tau) * (-1*self.firingrate + i)

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Synaptic drive of {self.name} : {firingrate}")

        return firingrate

    def build_HDC_to_HR_weight_matrix(self, weight_value = 0.3):
        w = np.identity(self.n)
        w = w * weight_value
        return w
    
    def build_HDC_to_I_weight_matrix(self, weight_value = 3):
        w = np.identity(self.n)
        w = w * weight_value
        return w
    

class VisualCells(Neurons):
    """ The VisualCells class will define a set of neurons that will fire in response to visual cues. 
    In the specific case of RatInTheBox, these neurons will fire when they see an object. Unlike ObjectVectorCells, 
    VisualCells cells provide no information on distance to a cue. Firing will be fixed until the object location changes.
    """
    default_params = {"n": 10, 
                      "name": "VisualCells",
                      "k_vis": 0.32,
                      "sigma_vis": 11} 

    def __init__(self,
                    Agent,
                    params={}): #<-- do not change these

        self.params = deepcopy(__class__.default_params) # to get the default param dictionary of the current class, defined in the preamble, use __class__. Then, make sure to deepcopy it, as only making a shallow copy can have unintended consequences (i.e., any modifications to it would be propagated to ALL instances of this class!).
        self.params.update(params)

        # Set any fixed variables for the neuron class
        self.N = self.params['n']
        self.name = self.params["name"]
        self.Agent = Agent
        self.k = self.params["k_vis"]
        self.sigma = self.params["sigma_vis"]

        # Calculate and store baseline weights
        var = (self.sigma ** 2) / (360 ** 2) * (self.N ** 2)
        p = (-0.5 * ((np.arange(self.N) - np.round(self.N/2)) ** 2)) / (var **2)
        W = self.k * np.exp(p)
        W =  deepcopy(np.roll(W, -1 * np.argmax(W)))

        self.zeroed_voltage = W

        super().__init__(Agent,self.params)

    def get_state(self, evaluate_at='agent', **kwargs): #<-- do not change these
        
        if "lights_on" in kwargs.keys() and ('lights_on' in kwargs.keys() and kwargs['lights_on']) and 'cue_location' in kwargs.keys():
            # The lights are on - we see the object and we can calculate firingrate
            
            if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Getting Synaptic drive of {self.name}")
         
            # get deepcopy of the zeroed_voltage
            visual_voltage = deepcopy(self.zeroed_voltage)
            
            firingrate = np.roll(visual_voltage, int(np.round(kwargs['cue_location'])))

        else:
            firingrate = np.zeros(self.N)  

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Synaptic drive of {self.name} : {firingrate}")
        
        return firingrate
    
    def build_V_to_HDC_weight_matrix(self, weight_value = 1, sigma = 11, N = 75):

        # var = (sigma ** 2) / (360 ** 2) * (N ** 2)
        # p = (-0.5 * ((np.arange(N) - np.round(N/2)) ** 2)) / (var ** 2)
        # w = weight_value * np.exp(p)
        # w =  deepcopy(np.roll(w, -1 * np.argmax(w)))

        w = np.identity(N)
        return w

class GainCells(Neurons):
    """ 
    The GainCells class will define a set of neurons that will include gain into the system.

    This will be built according to Ajabi et al 2023
    """
    default_params = {"n": 10, 
                      "name": "GainCells"} 

    def __init__(self,Agent,params={}): #<-- do not change these
        self.params = deepcopy(__class__.default_params) # to get the default param dictionary of the current class, defined in the preamble, use __class__. Then, make sure to deepcopy it, as only making a shallow copy can have unintended consequences (i.e., any modifications to it would be propagated to ALL instances of this class!).
        self.params.update(params)
        # use the parameters to setup the object itself
        self.n = self.params["n"]
        self.name = self.params["name"]
        super().__init__(Agent,self.params)


    def get_state(self,evaluate_at='agent',**kwargs): # <-- do not change these
        """ Get the firing rate of the current neuron, cycling through all its input layers
        If we want to add noise, it must be passed in the function (i.e. part of kwargs)
        """

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Getting Synaptic drive of {self.name}")
         

        if 'alpha' in kwargs.keys():
            a_in = kwargs['alpha']
        else:
            a_in = 1

        if 'verbose' in kwargs.keys() and kwargs['verbose']:
            print(f"a_in: {a_in}")
        
        firingrate = np.array(a_in - 1)

        
        if 'verbose' in kwargs.keys() and kwargs['verbose']:
                print(f"Synaptic drive of {self.name} : {firingrate}")
        
        return firingrate.T


    def build_G_to_HDC_weight_matrix(self, HDCs, weight_value = 4.8):
        
        w = np.identity(HDCs.n) * weight_value
        return w
