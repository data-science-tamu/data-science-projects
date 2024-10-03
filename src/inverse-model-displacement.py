import torch
import numpy as np

from sklearn import preprocessing

import time
from datetime import timedelta

from typing import Self # For nicer class return hints

# Program setup
state_messages = True
default_cpu = False # If false defaults to gpu

# Selecting device to use. If use_cpu is true, will default
# to cpu, otherwise will use GPU if a GPU is available

if default_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if state_messages: print("DEBUG: torch device is", device)

# Training information
learn_rate = 0.001
num_epochs = 100 # Each epoch is 1000 training "sessions"
data_shape = torch.tensor([256, 256], device=device)
# Where to find data (assumes follow same naming scheme as paper)
path_to_data = "./data"
trial_name = "m_z5_nu_z11"
output_folder = "./results"
sub_folders = True

# The model
class InverseModel(torch.nn.Module):
    # Changing the input and output sizes would require modifying the
    # loss function. 
    # NOTE: not the size of the entire data, but of a single vector entry
    INPUT_SIZE = 2 # [x, y]
    OUTPUT_SIZE = 2 # [pred_E, prev_v]
    
    # Model Configuration
    NUM_NEURON = 128 # Default Value
    NUM_HIDDEN_LAYERS = 16 # Default Value
    ACTIVATION_FUNCTION = torch.nn.ReLU
    
    # Effectively Enums below
    IS_STRAIN = 0
    IS_DISPLACEMENT = 1
    
    def __init__ (self, in_num = INPUT_SIZE, out_num = OUTPUT_SIZE, 
            num_neurons = NUM_NEURON, num_layers:int = NUM_HIDDEN_LAYERS,
            activation = ACTIVATION_FUNCTION, input_type = IS_STRAIN):
        super().__init__()
        
        self.input_type = input_type
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
    
    def convert_to_input_tensor(self, coordinates) -> torch.Tensor: # Uses input type
        if self.input_type == InverseModel.IS_STRAIN:
            return coordinates
        elif self.input_type == InverseModel.IS_DISPLACEMENT:
            return coordinates
        
        # TODO convert displacement coordinates to strain coordinates 
        # xn = (xn + xn+10)/2, reducing each dimension by one size
        
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
                b = (max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)
        
    def initialize_model(
            self, device_to=device, lr=learn_rate
        ) -> tuple[
            Self, torch.optim.Optimizer, torch.nn.modules.loss._Loss
            ]:
        self.to(device_to)
        self.apply(InverseModel.init_weight_and_bias)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_function = InverseLoss()
        
        return self, optimizer, loss_function
        
        
class InverseLoss(torch.nn.Module):
    E_CONSTRAINT = 0.25
    def __init__(self, mean_modulo=E_CONSTRAINT):
        super(InverseLoss, self).__init__()
        self.mean_modulo = mean_modulo
        
        # For loss calculation
        # To manipulate the pred_E (9)
        self.sum_kernel = torch.tensor(
            [[1.0, 1.0, 1.0], 
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],], 
            dtype=torch.float32, device=device)
        
        # Used to calculate partials of equilibrium condition
        self.wx_conv_xx = torch.tensor(
            [[-1.0, -1.0, -1.0], 
             [0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0], ],
            dtype = torch.float32, device=device)
        self.wx_conv_xy = torch.tensor(
            [[1.0, 0.0, -1.0], 
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0], ],
            dtype = torch.float32, device=device)
        self.wy_conv_yy = torch.tensor(
            [[1.0, 0.0, -1.0], 
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0], ],
            dtype = torch.float32, device=device)
        self.wy_conv_xy = torch.tensor(
            [[-1.0, -1.0, -1.0], 
             [0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0], ],
            dtype = torch.float32, device=device)