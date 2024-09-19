import torch

import numpy as np
import time

from sklearn import preprocessing

# Setup


# Resources Consulted for setting up the model
# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
# https://stackoverflow.com/a/49433937 # Weights

# Used Documentation
# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
# The number of in_features and out_features is the last dimension in their
# respective tensors.


class InverseModel(torch.nn.Module):
    # Default Values
    INPUT_SIZE = 2
    OUTPUT_SIZE = 1
    NUM_NEURON = 128
    NUM_HIDDEN_LAYERS = 16
    ACTIVATION_FUNCTION = torch.nn.ReLU
    
    LEARN_RATE = 0.001

    def __init__ (self, in_num = INPUT_SIZE, out_num = OUTPUT_SIZE, 
            num_neurons = NUM_NEURON, num_layers:int = NUM_HIDDEN_LAYERS,
            activation = ACTIVATION_FUNCTION):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.hidden1 = torch.nn.Linear(in_num, num_neurons)
        for i in range(2, num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(num_neurons, num_neurons))
        self.out = torch.nn.Linear(num_neurons, out_num)
        
        self.act1 = activation()
        for i in range(2, num_layers):
            setattr(self, f"act{i}", activation())
        self.act_out = activation()
    
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")( getattr(self, f"hidden{i}")(x) )
        x = self.act_out(self.out(x))
        return x

def init_weight_and_bias(layer):
    std_dev = 0.1
    max_dev = 2 # Maximum number of standard deviations from mean
    initial_bias = 0.1
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.trunc_normal_(
            layer.weight, 
            mean =0, 
            std = std_dev, 
            a = (-max_dev * std_dev), 
            b = (max_dev * std_dev)
        )
        layer.bias.data.fill_(initial_bias)
        
# Create the model (with default configuration)
model = InverseModel()
model.apply(init_weight_and_bias)
print(model)

def display_weights(model):
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            print("weights:", layer.state_dict()['weight'])
            print("bias:", layer.state_dict()['bias'])

display_weights(model)

quit()

# Below is incomplete code for training model, might move to a
# Jupyter Notebook for convenience and leave this file to contain just
# the model and a get model function

# Data imports for a given
path_to_data = "existing/elastnet"
trial_name = "m_z1_nu_z1"
disp_coord_data = np.loadtxt(f'{path_to_data}/data_compressible/{trial_name}/disp_coord')
disp_data_data = np.loadtxt(f'{path_to_data}/data_compressible/{trial_name}/disp_data')
m_data_data = np.loadtxt(f'{path_to_data}/data_compressible/{trial_name}/m_data')
nu_data_data = np.loadtxt(f'{path_to_data}/data_compressible/{trial_name}/nu_data')
strain_coord_data = np.loadtxt(f'{path_to_data}/data_compressible/{trial_name}/strain_coord')
strain_data_data = np.loadtxt(f'{path_to_data}/data_compressible/{trial_name}/strain_data')

# The inverse problem should use the disp_coord data as an input to the DNNs

# The given displacement data should be used to calculate loss and error.
# Ideally this could be done directly from the displacements, but only one
# (Axial?) is given. As such, this code uses the strain data to calculate
# the equilibrium conditions.
# Te offset nature of the strain data might be due to how it's discretized.
# The strain may be discretized in between the discrete coordinates of disp_coord. 
# Hence the dimensions being 1 less in length and the discretized coordinates 
# having a decimal (0.5)


# Standardize the inputs
# Reshape to guarantee the correct size. (-1 indicates any size)
ss_coordinates = preprocessing.StandardScaler()
disp_coord = ss_coordinates.fit_transform(disp_coord_data.reshape(-1, 2))

# Generate the initial values (random, but distributed according to a 
# truncated normal with standard deviation of 0.1)


