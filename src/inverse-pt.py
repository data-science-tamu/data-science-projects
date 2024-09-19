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
        

# Calculating the loss for the inverse model
"""
(12) L = w_r L_r + w_e L_e
where L_e is the elasticity loss,
and L_r is the residual force loss

(13) L_e = 1/(q^2) | sum(i=1 to q) sum(j=1 to q) E_pred(i,j) - sum(i=1 to q) sum(j=1 to q) E(i,j)|
where E_pred(x,y) is the elasticity at a given position (x, y).
and E(x,y) is the mean elasticity value (does it work with arbitrary value. ie: 0.5 or 1)
    It is a physical constraint
and q is the dimension of the elasticity image

A normalized MAE for the loss of the residual forces
(8) L_r = 1/(p^2) sum(i to p) sum(j to p) | e(i,j) | / (E_hat_pred (i, j))
Where e(i,j) is the residual force in a subregion
 and p the dimension of the residual force map
E_hat_pred(i,j) is the sum of the Young's Modulus values in a sub-region
Defined as
(9) sum(a=1 to 3) sum(b=1 to 3) E_pred(i+a-1, j+b-1)
- The calculation was done by sliding a 3 by 3 kernel of all ones.
"""
"""
From the provided elastnet.py

# For the strain:
def conv2d(x, W):
    return tf.nn.conv2d(
        tensor, convolution_matrix, strides=[1,1,1,1],padding='VALID'
    )
strain is calculated using conv 2d;
conv_x = [
    [-0.5, -0.5],
    [0.5, 0.5]]
conv_y = [
    [0.5, -0.5],
    [0.5, -0.5]]
u_mat = axial displacement? (u_x)
v_mat = lateral displacement? (u_y)

# equation (1)
e_xx = conv2d(u_mat, conv_x) # epsilon_xx
e_yy = conv2d(v_mat, conv_y) # epsilon_yy
r_xy = conv2d(u_mat, conv_y) + conv2d(v_mat, conv_x) # gamma_xy

Values are then adjusted
e_xx, e_yy, r_xy = 100 * reshape to 1D ([-1])
# Will likely just use the given strain data (no lateral? data given)
# for the inverse problem

# # For the elasticity
# # Elastic Constitutive Relation, equation (2)
# What the strain {} is multiplied by
ecr_matrix = ( 1 / (1 - v^2)) * [[ 1,  v,     0   ],
                                 [ v,  1,     0   ],
                                   0,  0, (1-v)/2 ]]
# Right side {e_xx, e_yy, r_xy}
strain_stack = stack([e_xx, e_yy, e_xy], axis = 1) # Or just the strain file

# The E value
y_mod_stack = stack([pred_m, pred_m, pred_m], axis = 1) # pred_m is the youngs modulus

# What to do with v, poisson's (especially the square) (guess below)
v_stack = stack([v_pred, v_pred, v_pred], axis = 1)
for the outside one, can do .multiply(v_stack, v_stack)

# The fraction in front.
=> nn.divide(y_mod_stack, 1 - nn.multiply(v_stack, v_stack))

# How to sub v into the matrix.
# Could try
ecr_matrices = [
    [[ 1,  v,     0   ],
     [ v,  1,     0   ],
       0,  0, (1-v)/2 ]], for v in v_stack[:, 0]
]
# ?
# need to test value results (with v = 0.5)


"""
class CustomLoss(torch.nn.Module):
    E_CONSTRAINT = 1
    def __init__(self):
        super().__init__()
        # The Losses
        self.loss_r = 1 
        self.loss_e = 1 

        # The constants multiples of loss
        self.weight_r = torch.nn.Parameter(0.25)
        self.weight_e = torch.nn.Parameter(0.25)

    def forward(self, x):
        return self.weight_r * self.loss_r + self.weight_e * self.loss_e

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


