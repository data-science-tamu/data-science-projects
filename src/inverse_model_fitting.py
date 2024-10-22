import torch
import numpy as np

from sklearn import preprocessing

import time
from datetime import timedelta

from typing import Tuple

# Program setup
STATE_MESSAGES = True
DEFAULT_CPU = False # If false defaults to gpu
DEFAULT_CPU = input("Default to CPU (y/n):").lower() == "y"
TENSOR_TYPE = torch.float32

# Selecting device to use. If use_cpu is true, will default
# to cpu, otherwise will use GPU if a GPU is available
# NOTE: When DEVICE is changed, call device refresh to move ELAS_OUTPUT_SHAPE
# to correct device
if DEFAULT_CPU:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if STATE_MESSAGES: print("STATE: torch device is", DEVICE)

# Training information, Default Values
NUM_FITTING_EPOCHS = 25
NUM_TRAINING_EPOCHS = 200 # Each epoch is 1000 training "sessions"
ITERATIONS_PER_EPOCH = 1000
NOTIFY_ITERATION_MOD = 100

FITTING_STRAIN = True
FITTING_DISPLACEMENT = False

# Output shape, assuming that displacement shape is 1 larger
# If data shape is (256, 256) then displacement is assumed at (257, 257)
ELAS_OUTPUT_SHAPE = torch.tensor([256, 256], device=DEVICE) 

# Model Information
LEARN_RATE = 0.001
# TODO: ask Dr. about using dropout layers.
# https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
# https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/
# Mentions a p of 0.2 is a good starting point

NUM_NEURON_FIT = 128
NUM_HIDDEN_LAYER_FIT = 16
ACTIVATION_FUNCTION_FIT = torch.nn.SiLU # aka "swish", the paper said it was best for displacement
ACTIVATION_FUNCTION_OUT_FIT = ACTIVATION_FUNCTION_FIT

NUM_NEURON_ELAS = 128
NUM_HIDDEN_LAYER_ELAS = 16
ACTIVATION_FUNCTION_ELAS = torch.nn.ReLU 
ACTIVATION_FUNCTION_OUT_ELAS = torch.nn.Softplus

# Where to find data (assumes follow same naming scheme as paper)
PATH_TO_DATA = "./data"
TRIAL_NAME = "m_z5_nu_z11"

# Output Path. NOTE: Must already be present in file system relative to where
# script is ran.
OUTPUT_FOLDER = "./results"
MODEL_SUBFOLDER = "/models"
PRE_FIT_MODEL_SUBFOLDER = "/pre_fitted"
LOSS_SUBFOLDER = "/loss" # TODO: Implement this
SAVE_LOSS = True # TRAIN: [x, y, e, d, total], FIT: [u]
                 # Weighted.

STRAIN_SIZE = 3 # [ε_xx, ε_yy, γ_xy] (epsilon/e, epsilon/e, gamma/r) # Strain Fit out
DISPLACEMENT_SIZE = 2 # [u_x, u_y] # DispFit out
COORDINATE_SIZE = 2 # [x, y] # Model in
ELAS_SIZE = 2 # [pred_E, pred_v] # Elas out

# Loss Parameters
E_CONSTRAINT = 0.25
WEIGHT_X = 1 # PDE equilibrium condition 1
WEIGHT_Y = 1 # PDE equilibrium condition 2
WEIGHT_E = 0.01
WEIGHT_D = 1.0

# ========================== Parameter Verification ========================== #

if (FITTING_STRAIN and FITTING_DISPLACEMENT):
    print("ERROR choose only one type to be fitting.")
    exit()

def refresh_devices():
    ELAS_OUTPUT_SHAPE.to(DEVICE)

# ================================== Models ================================== #
# Positional Encoding (asked for embedding but I found this instead, should be similar)
# TODO: refine and customize
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L222
import math
from torch import nn
class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

# Fitting Model
class FittingModel(torch.nn.Module):
    # Model Structure, parent of the different types of fitted inputs
    def __init__ (self, in_num, out_num):
        super().__init__()
        
        self.num_layers = NUM_HIDDEN_LAYER_FIT
        
        # Positional Embedding Layer for Fitting:
        # Currently Using Positional Encoding
        # https://www.vincentsitzmann.com/siren/
        # https://towardsdatascience.com/understanding-positional-encoding-in-transformers-dc6bafc021ab
        # TODO: refine and customize, currently only works for strain
        # Current version blows up memory usage. Try seeing if positionally encoding
        # The coordinates once, before passing to model works better.
        # This would also be nice as it would separate positional encoding
        # From the fitting itself.
        # Could make it a subfunction to make calling easier.
        self.positional_encoding = PosEncodingNeRF(in_features=in_num, sidelength=ELAS_OUTPUT_SHAPE)
        self.hidden1 = torch.nn.Linear(self.positional_encoding.out_dim, NUM_NEURON_FIT)
        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_FIT, NUM_NEURON_FIT))
        self.out = torch.nn.Linear(NUM_NEURON_FIT, out_num)
        
        self.act1 = ACTIVATION_FUNCTION_FIT()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", ACTIVATION_FUNCTION_FIT())
        self.act_out = ACTIVATION_FUNCTION_OUT_FIT()
    
    def forward(self, x):
        # x = self.positional_encoding(x).detach()
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")( getattr(self, f"hidden{i}")(x) )
        x = self.act_out(self.out(x))
        return x

    def init_weight_and_bias(layer):    
        std_dev = 0.1
        max_dev = 2 # Maximum number of standard deviations from mean
        mean = 0
        initial_bias = 0.1
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(
                layer.weight, 
                mean = mean, 
                std = std_dev, 
                a = (-max_dev * std_dev), 
                b = (max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)
    
class DisplacementFittingModel(FittingModel):
    # [u_x, u_y]
    def __init__(self):
        super().__init__(in_num=COORDINATE_SIZE, out_num=DISPLACEMENT_SIZE)

class StrainFittingModel(FittingModel):
    # [ε_xx, ε_yy, γ_xy] (epsilon/e, epsilon/e, gamma/r)
    def __init__(self):
        super().__init__(in_num=COORDINATE_SIZE, out_num=STRAIN_SIZE)
        
# Elasticity Model
class InverseModel(torch.nn.Module):
    # [E, v]
    def __init__ (self):
        super().__init__()
        
        self.num_layers = NUM_HIDDEN_LAYER_ELAS
        
        self.hidden1 = torch.nn.Linear(COORDINATE_SIZE, NUM_NEURON_ELAS)
        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_ELAS, NUM_NEURON_ELAS))
        self.out = torch.nn.Linear(NUM_NEURON_ELAS, ELAS_SIZE)
        
        self.act1 = ACTIVATION_FUNCTION_ELAS()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", ACTIVATION_FUNCTION_ELAS())
        self.act_out = ACTIVATION_FUNCTION_OUT_ELAS()
    
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")( getattr(self, f"hidden{i}")(x) )
        x = self.act_out(self.out(x))
        return x
    
    def init_weight_and_bias(layer):    
        std_dev = 0.1
        max_dev = 2 # Maximum number of standard deviations from mean
        mean = 0 # TODO: possibly increase this if nan issue persists
        initial_bias = 0.1
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(
                layer.weight, 
                mean = mean, 
                std = std_dev, 
                a = (-max_dev * std_dev), 
                b = (max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)


## =============================== Loss Function ============================== #
class InverseFittingLoss(torch.nn.Module):
    def __init__(self):
        super(InverseFittingLoss, self).__init__()

        # For loss calculation
        # Kernels used for strain calculations
        self.strain_conv_x = torch.nn.Conv2d(
            in_channels=1, # only one data point at the pixel in the image
            out_channels=1, 
            kernel_size=2, # 2 by 2 square
            bias=False,
            stride = 1,
            padding = 'valid') # No +- value added to the kernel values
        self.strain_conv_x.weight = torch.nn.Parameter(torch.tensor(
            [[-0.5, -0.5], 
             [ 0.5,  0.5]],
            dtype=TENSOR_TYPE, device=DEVICE).reshape(1, 1, 2, 2))
        
        self.strain_conv_y = torch.nn.Conv2d(
            in_channels=1, # only one data point at the pixel in the image
            out_channels=1, 
            kernel_size=2, # 2 by 2 square
            bias=False,
            stride = 1,
            padding = 'valid') # No +- value added to the kernel values
        self.strain_conv_y.weight = torch.nn.Parameter(torch.tensor(
            [[ 0.5, -0.5], 
             [ 0.5, -0.5]],
            dtype=TENSOR_TYPE, device=DEVICE).reshape(1, 1, 2, 2))
        
        # To manipulate the pred_E (9)
        self.sum_kernel = torch.tensor(
            [[1.0, 1.0, 1.0], 
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],], 
            dtype=TENSOR_TYPE, device=DEVICE)
        
        # Used to calculate partials of equilibrium condition
        self.wx_conv_xx = torch.tensor(
            [[-1.0, -1.0, -1.0], 
             [0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0], ],
            dtype = TENSOR_TYPE, device=DEVICE)
        self.wx_conv_xy = torch.tensor(
            [[1.0, 0.0, -1.0], 
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0], ],
            dtype = TENSOR_TYPE, device=DEVICE)
        self.wy_conv_yy = torch.tensor(
            [[1.0, 0.0, -1.0], 
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0], ],
            dtype = TENSOR_TYPE, device=DEVICE)
        self.wy_conv_xy = torch.tensor(
            [[-1.0, -1.0, -1.0], 
             [0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0], ],
            dtype = TENSOR_TYPE, device=DEVICE)

    def forward(self, fitted_data:torch.Tensor, experimental_data:torch.Tensor,
                elas_output:torch.Tensor=None, only_fitting=False, loss_array:list=None, 
                save_loss_condition=False):
        if (FITTING_DISPLACEMENT):
            # Equation (10), modified
            pred_displacement = fitted_data
            loss_d = torch.mean(torch.abs(pred_displacement-experimental_data))
        
            if only_fitting:
                if SAVE_LOSS and save_loss_condition:
                    loss_array.append(loss_d)
                return loss_d # * InverseFittingLoss.WEIGHT_U # Shouldn't need the weight here
            
            strain = self.calculate_strain(displacement_stack=pred_displacement)
            
        elif FITTING_STRAIN:
            # Equation (10), modified
            strain = fitted_data
            loss_d = torch.mean(torch.abs(strain-experimental_data))
        
            if only_fitting:
                if SAVE_LOSS and save_loss_condition:
                    loss_array.append(loss_d)
                return loss_d # * InverseFittingLoss.WEIGHT_U # Shouldn't need the weight here

        pred_E = elas_output[:,0]
        pred_v = elas_output[:,1]
        stress = self.calculate_stress(pred_E, pred_v, strain)
        
        # DEBUG, it seems that predE and stress(calculated from displacements)
        # Are occasionally optimized to 0, causing this loss_r function to return nan.
        loss_x, loss_y = self.calculate_loss_r(pred_E, stress)
        
        # Equation (13)
        loss_e = torch.abs(torch.mean(pred_E) - E_CONSTRAINT)
        
        if SAVE_LOSS and save_loss_condition:
            loss_array.append([
                WEIGHT_X * loss_x.item(),
                WEIGHT_Y * loss_y.item(), 
                WEIGHT_E * loss_e.item(), 
                WEIGHT_D * loss_d.item(),
                WEIGHT_X * loss_x + WEIGHT_Y * loss_y + loss_e*WEIGHT_E + loss_d * WEIGHT_D
            ])
            
        # Modified equation (12) with data loss (10). 
        return WEIGHT_X * loss_x + WEIGHT_Y * loss_y + loss_e*WEIGHT_E + loss_d * WEIGHT_D

    # This function takes an unprocessed displacement (raw data of shape [:, 2])
    # Assuming each entry is [u_x, u_y] ([u, v] from papers code)
    def calculate_strain(self, displacement_stack:torch.Tensor) -> torch.Tensor:

        ux_matrix = displacement_stack[:, 0].reshape(1, 1, ELAS_OUTPUT_SHAPE[0]+1, ELAS_OUTPUT_SHAPE[1]+1)
        uy_matrix = displacement_stack[:, 1].reshape(1, 1, ELAS_OUTPUT_SHAPE[0]+1, ELAS_OUTPUT_SHAPE[1]+1)

        e_xx = self.strain_conv_x(ux_matrix) # u_xx
        e_yy = self.strain_conv_y(uy_matrix) # u_yy
        r_xy = self.strain_conv_y(ux_matrix) + self.strain_conv_x(uy_matrix) # u_xy + u_yx
        
        # The following is from the paper, 
        # NOTE: I don't know why it is multiplied by 100
        e_xx = 100*e_xx.reshape(-1)
        e_yy = 100*e_yy.reshape(-1)
        r_xy = 100*r_xy.reshape(-1)
        
        strain = torch.stack([e_xx, e_yy, r_xy], dim=1)
        return strain
    
    # Based on equation(2), the elastic constitutive relation
    def calculate_stress(self, pred_E:torch.Tensor, pred_v:torch.Tensor, strain:torch.Tensor):
        # Strain comes stacked (each row is (e_xx, e_yy, r_xy))
        E_stack = torch.stack([pred_E, pred_E, pred_E], dim=1)
        v_stack = torch.stack([pred_v, pred_v, pred_v], dim=1)
        
        # Create a stack of c_matrices from the predicted v's
        c_stack = torch.stack([
            torch.ones(pred_v.shape, dtype=torch.float32, device=DEVICE),
            pred_v,
            torch.zeros(pred_v.shape, dtype=torch.float32, device=DEVICE), ##
            pred_v,
            torch.ones(pred_v.shape, dtype=torch.float32, device=DEVICE),
            torch.zeros(pred_v.shape, dtype=torch.float32, device=DEVICE), ##
            torch.zeros(pred_v.shape, dtype=torch.float32, device=DEVICE),
            torch.zeros(pred_v.shape, dtype=torch.float32, device=DEVICE),
            torch.divide(
                (torch.ones(pred_v.shape, dtype=torch.float32, device=DEVICE) - pred_v),
                torch.full(pred_v.shape, 2.0, dtype=torch.float32, device=DEVICE)
            ) ##
        ], dim=1).reshape([-1, 3,3])
        
        # bmm = batch matrix multiplication. Essentially for each item in both
        # matrix multiply them. The squeeze is remove dimensions of size
        # 1 resulting from the strict matrix multiplication.(1x3)(3x3)
        matmul_results = torch.bmm(strain.reshape(-1,1,3), c_stack).squeeze()
        
        v2 = torch.square(v_stack)
        fraction = torch.divide(E_stack, 1 - v2)
        stress = torch.multiply(matmul_results, fraction)
        return stress
    
    # Based on equation (8)
    # NOTE: see if switching to using nn.conv2d is better. (Implicitly calls
    # the functional version, but has a gradient which might help with learning)
    def calculate_loss_r(self, pred_E, stress):
        def conv2d(x, W:torch.Tensor):
            W = W.view(1,1,3,3).repeat(1,1,1,1)
            return torch.nn.functional.conv2d(
                x, W,
                stride = 1,
                padding = 'valid'
            )
        
        # Young's modulus. (E_hat from equation (9))
        # Kernel moved to initialization to avoid repeating step
        
        pred_E_matrix = torch.reshape(pred_E, [ELAS_OUTPUT_SHAPE[0], ELAS_OUTPUT_SHAPE[1]])
        pred_E_matrix_4d = torch.reshape(pred_E_matrix, [-1, 1, ELAS_OUTPUT_SHAPE[0], ELAS_OUTPUT_SHAPE[1]])
        pred_E_conv = conv2d(pred_E_matrix_4d, self.sum_kernel)
        
        # Unstack the stress
        stress_xx = stress[:, 0]
        stress_yy = stress[:, 1]
        stress_xy = stress[:, 2]
        
        stress_xx_matrix_4d = torch.reshape(stress_xx, [-1, 1, ELAS_OUTPUT_SHAPE[0], ELAS_OUTPUT_SHAPE[1]])
        stress_yy_matrix_4d = torch.reshape(stress_yy, [-1, 1, ELAS_OUTPUT_SHAPE[0], ELAS_OUTPUT_SHAPE[1]])
        stress_xy_matrix_4d = torch.reshape(stress_xy, [-1, 1, ELAS_OUTPUT_SHAPE[0], ELAS_OUTPUT_SHAPE[1]])
        
        # Convolutions for derivatives
        # Moved to initialization step to avoid repeatedly making new tensors.

        # From equilibrium condition
        fx_conv_xx = conv2d(stress_xx_matrix_4d, self.wx_conv_xx)
        fx_conv_xy = conv2d(stress_xy_matrix_4d, self.wx_conv_xy)
        fx_conv_sum = fx_conv_xx + fx_conv_xy # Result that should be 0

        fy_conv_yy = conv2d(stress_yy_matrix_4d, self.wy_conv_yy)
        fy_conv_xy = conv2d(stress_xy_matrix_4d, self.wy_conv_xy)
        fy_conv_sum = fy_conv_yy + fy_conv_xy # Result that should be 0

        # Normalization, doing equation (8), rest was calculating residual forces
        fx_conv_sum_norm = torch.divide(fx_conv_sum, pred_E_conv)
        fy_conv_sum_norm = torch.divide(fy_conv_sum, pred_E_conv)
        
        # Finally get value of loss
        loss_x = torch.mean(torch.abs(fx_conv_sum_norm))
        loss_y = torch.mean(torch.abs(fy_conv_sum_norm))
        return loss_x, loss_y


## =============================== Runner Class ============================== #
class InverseFittingRunner():
    def print_gpu_memory():
        if STATE_MESSAGES and DEVICE.type == 'cuda':
            print(torch.cuda.get_device_name(device=DEVICE))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(device=DEVICE)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(device=DEVICE)/1024**3,1), 'GB')

    def coordinates_to_input_tensor(coordinates:np.ndarray, device_to=DEVICE) -> torch.Tensor:
        ss_coordinates = preprocessing.StandardScaler()
        return torch.tensor(
            ss_coordinates.fit_transform(coordinates.reshape(-1, 2)),
            dtype=torch.float32,
            device=device_to)

    # Coordinate array should have dim = 1
    def disp_to_elas_input_tensor(coordinates:np.ndarray, 
                                device_to=DEVICE) -> torch.Tensor: # Uses input type
        # Create Convolution        
        coord_conv = torch.nn.Conv2d(
            in_channels=1, # only one data point at the pixel in the image
            out_channels=1, 
            kernel_size=2, # 2 by 2 square
            bias=False,
            stride = 1,
            padding = 'valid') # No +- value added to the kernel values
        coord_conv.weight = torch.nn.Parameter(torch.tensor(
            [[ 0.25,  0.25], 
            [ 0.25,  0.25]], dtype=TENSOR_TYPE, device=DEVICE).reshape(1, 1, 2, 2))
        
        # Create image from coordinates. +1 as displacement coordinates have 1
        # more dimensional size than output elasticity due to an applied convolution
        coord_mat = torch.tensor(coordinates.reshape(ELAS_OUTPUT_SHAPE[0]+1, ELAS_OUTPUT_SHAPE[1]+1, 2),
                                    dtype=TENSOR_TYPE, device=DEVICE)
        
        coord_mat_x = coord_mat[:,:,0]
        result_x = coord_conv(coord_mat_x.reshape(1,1,ELAS_OUTPUT_SHAPE[0]+1,ELAS_OUTPUT_SHAPE[1]+1)).reshape(-1)
        
        coord_mat_y = coord_mat[:,:,1]
        result_y = coord_conv(coord_mat_y.reshape(1,1,ELAS_OUTPUT_SHAPE[0]+1,ELAS_OUTPUT_SHAPE[1]+1)).reshape(-1)
        strain_coordinates = torch.stack([result_x, result_y], dim=1).detach()
        
        return InverseFittingRunner.coordinates_to_input_tensor(strain_coordinates.cpu(), device_to)
 
    def save_elasticity_model_eval(model:InverseModel, coordinate_tensor, file_name_E, file_name_v) -> None:
        model.eval()
        output = model(coordinate_tensor)
        # assert isinstance(output, torch.Tensor) # For intellisense typing convenience
        pred_E = output[:, 0]
        pred_v = output[:, 1]
        np.savetxt(file_name_E, pred_E.cpu().detach().numpy())
        np.savetxt(file_name_v, pred_v.cpu().detach().numpy())
        
    def save_displacement_fit_eval(model:DisplacementFittingModel, coordinate_tensor:torch.Tensor, 
                        file_name_ux, file_name_uy) ->None:
        model.eval()
        output = model(coordinate_tensor)
        # assert isinstance(output, torch.Tensor) # For intellisense typing convenience
        pred_ux = output[:, 0]
        pred_uy = output[:, 1]
        np.savetxt(file_name_ux, pred_ux.cpu().detach().numpy())
        np.savetxt(file_name_uy, pred_uy.cpu().detach().numpy())

    def save_strain_fit_eval(model:StrainFittingModel, coordinate_tensor:torch.Tensor, 
                        file_name_exx, file_name_eyy, file_name_rxy) ->None:
        model.eval()
        output = model(coordinate_tensor)
        # assert isinstance(output, torch.Tensor) # For intellisense typing convenience
        pred_exx = output[:, 0]
        pred_eyy = output[:, 1]
        pred_rxy = output[:, 2]
        np.savetxt(file_name_exx, pred_exx.cpu().detach().numpy())
        np.savetxt(file_name_eyy, pred_eyy.cpu().detach().numpy())
        np.savetxt(file_name_rxy, pred_rxy.cpu().detach().numpy())

    def initialize_and_train_disp(
            # WARNING: Many of these are different from initialize_and_train_model.
            # Requiring the final tensors instead of input numpy arrays.
            disp_model:DisplacementFittingModel, 
            loss_function:InverseFittingLoss,
            disp_coord:torch.Tensor, 
            displacement_data:torch.Tensor,
            fit_new_model = False,
        ) -> None: # Modifies models inplace
        disp_model_path  = f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}{PRE_FIT_MODEL_SUBFOLDER}/" # Path
        disp_model_path += f"displacement_fit_{TRIAL_NAME}_{NUM_FITTING_EPOCHS}e.pt" # Name
        
        # Initialize Displacement Model on Correct Device
        disp_model.to(DEVICE)
        
        # Check if current model exists
        if not fit_new_model:
            try:
                disp_model.load_state_dict( torch.load(
                        disp_model_path, map_location=DEVICE, weights_only=True
                ))
                return disp_model
            except:
                pass  # On failure, assumed file not found, train model
        
        # ============ Fit model if pre-trained model not found. ============         
        # Initialize Model Weights
        disp_model.apply(DisplacementFittingModel.init_weight_and_bias)
        
        # Initialize Optimizer
        optimizer = torch.optim.Adam(disp_model.parameters(), lr=LEARN_RATE)
        
        if STATE_MESSAGES: print("STATE: starting displacement fitting")
        
        loss_array = []
        
        training_start_time=time.time()
        for e in range(NUM_FITTING_EPOCHS):
            print(f"Fitting Epoch {e} starting")
            
            disp_model.train()
            epoch_start_time = time.time()
            for i in range (1, 1001):
                optimizer.zero_grad() # Resets the optimizer?
                output_disp = disp_model(disp_coord)
                loss = loss_function(
                    fitted_data = output_disp,
                    experimental_data=displacement_data,
                    only_fitting=True,
                    loss_array=loss_array,
                    save_loss_condition = i % NOTIFY_ITERATION_MOD == 0
                )
                loss.backward()
                optimizer.step()
                if i % NOTIFY_ITERATION_MOD == 0:
                        print(f"Fitting Epoch: {e} [{i}/1000 ({i/10.0:.2f}%)]\tLoss: {loss.item():.6f}")
            
            e_time = time.time()-epoch_start_time
            print(f"Epoch{e} took {e_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(NUM_FITTING_EPOCHS-e) * e_time)}")
            
            InverseFittingRunner.save_displacement_fit_eval(
                disp_model, disp_coord,
                f"{OUTPUT_FOLDER}/pred_ux/fit{e}.txt",
                f"{OUTPUT_FOLDER}/pred_uy/fit{e}.txt"
            )
        
        if SAVE_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}{LOSS_SUBFOLDER}/{TRIAL_NAME}_fitting_loss.txt", torch.Tensor(loss_array).cpu().detach().numpy())
        
        torch.save(disp_model.state_dict(), disp_model_path)

    def initialize_and_train_strain(
            # WARNING: Many of these are different from initialize_and_train_model.
            # Requiring the final tensors instead of input numpy arrays! 
            strain_model:StrainFittingModel, 
            loss_function:InverseFittingLoss,
            strain_coord:torch.Tensor, 
            strain_data:torch.Tensor,
            fit_new_model = False,
        ) -> None: # Modifies models inplace
        strain_model_path  = f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}{PRE_FIT_MODEL_SUBFOLDER}/" # Path
        strain_model_path += f"strain_fit_{TRIAL_NAME}_{NUM_FITTING_EPOCHS}e.pt" # Name
        
        # Initialize Displacement Model on Correct Device
        strain_model.to(DEVICE)
                
        # TODO: REMOVE THIS AS JUST USING FOR TESTING IDEA
        strain_model_path += ".test"
        pos_encode = PosEncodingNeRF(2, [256,256]).to(DEVICE)
        strain_coord = pos_encode(strain_coord.reshape([256,256])).reshape([-1,pos_encode.out_dim])
        
        # Check if current model exists
        if not fit_new_model:
            try:
                strain_model.load_state_dict( torch.load(
                        strain_model_path, map_location=DEVICE, weights_only=True
                ))
                return strain_model
            except:
                pass  # On failure, assumed file not found, train model
        
        # ============ Fit model if pre-trained model not found. ============         
        # Initialize Model Weights
        strain_model.apply(DisplacementFittingModel.init_weight_and_bias)
        
        # Initialize Optimizer
        optimizer = torch.optim.Adam(strain_model.parameters(), lr=LEARN_RATE)
        
        if STATE_MESSAGES: print("STATE: starting strain fitting")
        
        loss_array = []
        
        training_start_time=time.time()
        for e in range(NUM_FITTING_EPOCHS):
            print(f"Fitting Epoch {e} starting")
            
            strain_model.train()
            epoch_start_time = time.time()
            for i in range (1, 1001):
                optimizer.zero_grad() # Resets the optimizer?
                fitted_strain = strain_model(strain_coord)
                loss = loss_function(
                    fitted_data = fitted_strain,
                    experimental_data = strain_data,
                    only_fitting = True,
                    loss_array=loss_array,
                    save_loss_condition = i % NOTIFY_ITERATION_MOD == 0
                )
                loss.backward()
                optimizer.step()
                if i % NOTIFY_ITERATION_MOD == 0:
                        print(f"Fitting Epoch: {e} [{i}/1000 ({i/10.0:.2f}%)]\tLoss: {loss.item():.6f}")
            e_time = time.time()-epoch_start_time
            
            print(f"Epoch{e} took {e_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(NUM_FITTING_EPOCHS-e) * e_time)}")
            
                        
            
            InverseFittingRunner.save_strain_fit_eval(
                strain_model, strain_coord,
                f"{OUTPUT_FOLDER}/pred_exx/fit{e}.txt",
                f"{OUTPUT_FOLDER}/pred_eyy/fit{e}.txt",
                f"{OUTPUT_FOLDER}/pred_rxy/fit{e}.txt"
            )
        
        if SAVE_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}{LOSS_SUBFOLDER}/{TRIAL_NAME}_fitting_loss.txt", torch.Tensor(loss_array).cpu().detach().numpy())
        
        torch.save(strain_model.state_dict(), strain_model_path)

    def initialize_and_train_elas(
            # WARNING: Many of these are different from initialize_and_train_model.
            # Requiring the final tensors instead of input numpy arrays! 
            elas_model:InverseModel,
            fitting_model:FittingModel, 
            loss_function:InverseFittingLoss,
            elas_coord:torch.Tensor, 
            fit_coord:torch.Tensor,
            data_to_fit:torch.tensor,
        ) -> None: # Modifies models inplace
        # TODO: try with a fixed displacement after fitting (consider both scenarios this and different loss terms
            # Since the boundary values seem to be off, could do loss without them (as they are inaccurate). 
            # Or more iterations of fitting.
        
        # Initialize elasticity model on correct device
        elas_model.to(DEVICE)
        elas_model.apply(InverseModel.init_weight_and_bias)
        
        # Start Training
        if STATE_MESSAGES: print("STATE: training for elasticity constants")
        optimizer = torch.optim.Adam(
            list(fitting_model.parameters()) + list(elas_model.parameters()), 
            lr=LEARN_RATE)
        # NOTE: depending on the the test, including or don't include disp_model.parameters (when not using loss_u)
        
        loss_array = []
                    
        training_start_time=time.time()
        for e in range(NUM_TRAINING_EPOCHS):
            print(f"Training Epoch {e} starting")
            fitting_model.train()
            epoch_start_time = time.time()
            for i in range (1, 1001):
                optimizer.zero_grad() # Resets the optimizer?
                fitted_data = fitting_model(fit_coord)
                elas_output = elas_model(elas_coord)
                loss = loss_function.forward(
                    fitted_data = fitted_data,
                    elas_output = elas_output,
                    experimental_data = data_to_fit,
                    only_fitting=False,
                    loss_array=loss_array,
                    save_loss_condition = i % NOTIFY_ITERATION_MOD == 0
                )
                loss.backward()
                optimizer.step()
                if i % NOTIFY_ITERATION_MOD == 0:
                    print(f"Training Epoch: {e} [{i}/1000 ({i/10.0:.2f}%)]\tLoss: {loss.item():.6f}")
            
            e_time = time.time()-epoch_start_time
            print(f"Epoch{e} took {e_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(NUM_TRAINING_EPOCHS-e) * e_time)}")
            
            if FITTING_DISPLACEMENT:
                InverseFittingRunner.save_elasticity_model_eval(
                    elas_model, elas_coord,
                    f"{OUTPUT_FOLDER}/pred_E/epoch{e}.txt",
                    f"{OUTPUT_FOLDER}/pred_v/epoch{e}.txt"
                )
                InverseFittingRunner.save_displacement_fit_eval(
                    fitting_model, fit_coord,
                    f"{OUTPUT_FOLDER}/pred_ux/epoch{e}.txt",
                    f"{OUTPUT_FOLDER}/pred_uy/epoch{e}.txt"
                )
            elif FITTING_STRAIN:
                InverseFittingRunner.save_elasticity_model_eval(
                    elas_model, elas_coord,
                    f"{OUTPUT_FOLDER}/pred_E/epoch{e}.txt",
                    f"{OUTPUT_FOLDER}/pred_v/epoch{e}.txt"
                )
                InverseFittingRunner.save_strain_fit_eval(
                    fitting_model, fit_coord,
                    f"{OUTPUT_FOLDER}/pred_exx/epoch{e}.txt",
                    f"{OUTPUT_FOLDER}/pred_eyy/epoch{e}.txt",
                    f"{OUTPUT_FOLDER}/pred_rxy/epoch{e}.txt",
                )
        
        if SAVE_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}{LOSS_SUBFOLDER}/{TRIAL_NAME}_training_loss.txt", torch.Tensor(loss_array).cpu().detach().numpy())
        
        if STATE_MESSAGES: print("STATE: Training Finished")

    def initialize_and_train_model(
            elas_model:InverseModel,
            fitting_model:FittingModel, 
            coordinate_array: np.ndarray, 
            data_array_to_fit:np.ndarray,
            fit_new_model = False,
        ) -> None:
        # Standardize the Discrete Coordinates into a Tensor
        fit_coord = InverseFittingRunner.coordinates_to_input_tensor(
            coordinate_array, device_to=DEVICE
        )
        # Create Displacement Data Tensor from numpy array
        data_to_fit = torch.tensor(
            data_array_to_fit, 
            dtype=TENSOR_TYPE,
            device=DEVICE)

        # Initialize Shared Loss Function
        loss_function = InverseFittingLoss()
        
        if FITTING_DISPLACEMENT:
            # For Elasticity, Reduce Dimension Size as Elasticity Calculation Reduces Dimensions
            elas_coord = InverseFittingRunner.disp_to_elas_input_tensor(
                coordinate_array, device_to=DEVICE
            )
            InverseFittingRunner.initialize_and_train_disp(
                disp_model = fitting_model,
                loss_function = loss_function,
                disp_coord = fit_coord,
                displacement_data = data_to_fit,
                fit_new_model = fit_new_model,
            )
            
        elif FITTING_STRAIN:
            elas_coord = fit_coord
            InverseFittingRunner.initialize_and_train_strain(
                strain_model = fitting_model,
                loss_function = loss_function,
                strain_coord = fit_coord,
                strain_data = data_to_fit,
                fit_new_model = fit_new_model,
            )
        
        InverseFittingRunner.initialize_and_train_elas(
            elas_model = elas_model,
            fitting_model = fitting_model,
            loss_function = loss_function,
            elas_coord = elas_coord,
            fit_coord = fit_coord,
            data_to_fit = data_to_fit,
        )

def main() -> None:
    global NUM_TRAINING_EPOCHS
    global NUM_FITTING_EPOCHS
    NUM_FITTING_EPOCHS = 2
    NUM_TRAINING_EPOCHS = 2
    
    global FITTING_STRAIN
    global FITTING_DISPLACEMENT
    FITTING_STRAIN = True
    FITTING_DISPLACEMENT = False
    
    global WEIGHT_D
    WEIGHT_D = 1.0
    
    # LOAD Data. Called data_data sometimes as file is called _data.
    disp_coord_array= np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/disp_coord')
    disp_data_array= np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/disp_data')
    m_data_array= np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/m_data')
    nu_data_array= np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/nu_data')
    strain_coord_array= np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/strain_coord')
    strain_data_array= np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/strain_data')
    
    if STATE_MESSAGES: print("STATE: data imported")

    elas_model = InverseModel()
    strain_model = StrainFittingModel()
    InverseFittingRunner.initialize_and_train_model(
        elas_model = elas_model,
        fitting_model = strain_model,
        coordinate_array = strain_coord_array,
        data_array_to_fit = strain_data_array,
    )
       
    # Save DNN
    torch.save(elas_model.state_dict(), 
        f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/elasticity_{TRIAL_NAME}_{WEIGHT_D}wd_{NUM_FITTING_EPOCHS}f_{NUM_TRAINING_EPOCHS}e.pt")
    torch.save(strain_model.state_dict(),
        f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/strain_fit_{TRIAL_NAME}_{WEIGHT_D}wd_{NUM_FITTING_EPOCHS}f_{NUM_TRAINING_EPOCHS}e.pt")
    
    if STATE_MESSAGES: print("STATE: Done")

if __name__ == "__main__":
    main()
