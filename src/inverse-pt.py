import torch

import numpy as np
import time

from sklearn import preprocessing

# Setup
num_neuron = 128
learn_rate = 0.001

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

def weight_variable(shape):
    std_dev = 0.1
    max_dev = 2 # Maximum number of standard deviations from mean
    w = torch.empty(shape)
    torch.nn.init.trunc_normal_(
        w, 
        mean=0, 
        std=std_dev, 
        a= -max_dev*std_dev, 
        b= max_dev*std_dev
    )
    return torch.nn.Parameter(w)

def bias_variable(shape):
    initial_bias = 0.1
    b = torch.const
    
    return torch.nn.Parameter(b)