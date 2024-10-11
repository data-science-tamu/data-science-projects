import torch
import numpy as np

from sklearn import preprocessing

import time
from datetime import timedelta

from typing import Tuple

# Program setup
STATE_MESSAGES = True
DEFAULT_CPU = False # If false defaults to gpu
TENSOR_TYPE = torch.float32

# Selecting device to use. If use_cpu is true, will default
# to cpu, otherwise will use GPU if a GPU is available

if DEFAULT_CPU:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if STATE_MESSAGES: print("DEBUG: torch device is", DEVICE)

# Training information, Default Values
LEARN_RATE = 0.001
NUM_EPOCHS = 200 # Each epoch is 1000 training "sessions"
NUM_FITTING_ONLY_EPOCHS = 25

# Output shape, assuming that displacement shape is 1 larger
# If data shape is (256, 256) then displacement is assumed at (257, 257)
DATA_SHAPE = torch.tensor([256, 256], device=DEVICE) 

# Where to find data (assumes follow same naming scheme as paper)
PATH_TO_DATA = "./data"
TRIAL_NAME = "m_z5_nu_z11"
OUTPUT_FOLDER = "./results"
SUB_FOLDERS = True

# The Displacement Fitting Model
class DisplacementFitModel(torch.nn.Module):
    # Changing the input and output sizes would require modifying the
    # loss function. 
    # NOTE: not the size of the entire data, but of a single vector entry
    INPUT_SIZE = 2 # [x, y]
    OUTPUT_SIZE = 2 # [pred_ux, prev_uy]
    
    # Model Configuration
    NUM_NEURON = 128 # Default Value
    NUM_HIDDEN_LAYERS = 16 # Default Value
    ACTIVATION_FUNCTION = torch.nn.SiLU # aka "swish", the paper said it was best for displacement
    
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
                b = (max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)


# The Elasticity Model
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
                b = (max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)


# Loss function for displacement fitting case
class InverseFittingLoss(torch.nn.Module):
    E_CONSTRAINT = 0.25 # A randomly chosen default value
    # It was 0.01 in the paper, but my first value was then 0.000919
    # TODO: see if the weights work 
    WEIGHT_U = 1
    WEIGHT_E = 0.01
    
    
    def __init__(self, mean_modulo=E_CONSTRAINT):
        super(InverseFittingLoss, self).__init__()
        self.mean_modulo = mean_modulo
        
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
    
    def forward(self, pred_ux, pred_uy, displacement_data:torch.Tensor,
                pred_E:torch.Tensor=None, pred_v:torch.Tensor=None, # For when only fitting
                only_fitting=False):
        # Equation (10), modified # TODO: ask prof about correctness
        pred_displacement = torch.stack([pred_ux, pred_uy], dim=1)
        loss_u = torch.mean(torch.abs(pred_displacement-displacement_data))
        
        if only_fitting:
            return loss_u * InverseFittingLoss.WEIGHT_U
        
        strain = self.calculate_strain(displacement_stack=pred_displacement)
        
        stress = self.calculate_stress(pred_E, pred_v, strain)
        loss_x, loss_y = self.calculate_loss_r(pred_E, stress)
        
        # Equation (13)
        loss_e = torch.abs(torch.mean(pred_E) - self.mean_modulo)
        
        # Modified equation (12) with data loss (10). 
        # 1/100 is the value used in the given code, I don't know it's origin/reason.
        # TODO: occasionally display the values to see the scale.
        # loss_e shouldn't be very important (just making sure e isn't 0), if it goes to 0 increase WEIGHT_E
        # pde loss should be most important
        return loss_x + loss_y + loss_e * InverseFittingLoss.WEIGHT_E + loss_u * InverseFittingLoss.WEIGHT_U
    
    # Probably better to call this once instead of upon every iteration
    # Seeing as it is static (in so much as it doesn't use the predicted
    # Elasticity values.)
    # This function takes an unprocessed displacement (raw data of shape [:, 2])
    # Assuming each entry is [u_x, u_y] ([u, v] from papers code)
    def calculate_strain(self, displacement_stack:torch.Tensor) -> torch.Tensor:

        ux_matrix = displacement_stack[:, 0].reshape(1, 1, DATA_SHAPE[0]+1, DATA_SHAPE[1]+1)
        uy_matrix = displacement_stack[:, 1].reshape(1, 1, DATA_SHAPE[0]+1, DATA_SHAPE[1]+1)

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
    def calculate_stress(self, pred_E, pred_v, strain:torch.Tensor):
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
        
        # Significantly faster than using list comprehension.
        # Likely due to optimize batching and possibly involving the cpu less
        # The only real difference is the need to squeeze (remove the dimensions=1)
        # As the output would otherwise be a vector of 1x3 matrices and we want
        # A vector of vectors in R3.
        matmul_results = torch.bmm(strain.reshape(-1,1,3), c_stack).squeeze()
        
        v2 = torch.square(v_stack)
        fraction = torch.divide(E_stack, 1 - v2)
        stress = torch.multiply(matmul_results, fraction)
        return stress
    
    # Based on equation (8)
    # TODO: see if switching to using nn.conv2d is better. (Implicitly calls
    # the functional version, but has a gradient which might help with learning)
    def calculate_loss_r(self, pred_E, stress) -> tuple[float, float]:
        def conv2d(x, W:torch.Tensor):
            W = W.view(1,1,3,3).repeat(1,1,1,1)
            return torch.nn.functional.conv2d(
                x, W,
                stride = 1,
                padding = 'valid'
            )
        
        # Young's modulus. (E_hat from equation (9))
        # Kernel moved to initialization to avoid repeating step
        
        pred_E_matrix = torch.reshape(pred_E, [DATA_SHAPE[0], DATA_SHAPE[1]])
        pred_E_matrix_4d = torch.reshape(pred_E_matrix, [-1, 1, DATA_SHAPE[0], DATA_SHAPE[1]])
        pred_E_conv = conv2d(pred_E_matrix_4d, self.sum_kernel)
        
        # Unstack the stress
        stress_xx = stress[:, 0]
        stress_yy = stress[:, 1]
        stress_xy = stress[:, 2]
        
        stress_xx_matrix_4d = torch.reshape(stress_xx, [-1, 1, DATA_SHAPE[0], DATA_SHAPE[1]])
        stress_yy_matrix_4d = torch.reshape(stress_yy, [-1, 1, DATA_SHAPE[0], DATA_SHAPE[1]])
        stress_xy_matrix_4d = torch.reshape(stress_xy, [-1, 1, DATA_SHAPE[0], DATA_SHAPE[1]])
        
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
    

# Contains code to setup and train the case of an inverse model with data fitting.
class InverseFittingRunner():
    def print_gpu_memory():
        if STATE_MESSAGES and DEVICE.type == 'cuda':
            print(torch.cuda.get_device_name(device=DEVICE))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(device=DEVICE)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(device=DEVICE)/1024**3,1), 'GB')
    
    def _initialize_model(
                elasticity_model:InverseModel, 
                displacement_model:DisplacementFitModel,
                device_to=DEVICE, 
                lr=LEARN_RATE
            ) -> Tuple[
                    InverseModel, DisplacementFitModel, torch.optim.Optimizer, InverseFittingLoss
                ]:
        elasticity_model.to(device_to)
        elasticity_model.apply(InverseModel.init_weight_and_bias)
        displacement_model.to(device_to)
        displacement_model.apply(DisplacementFitModel.init_weight_and_bias)
        optimizer = torch.optim.Adam(
            list(elasticity_model.parameters()) + list(displacement_model.parameters()), 
            lr=lr) # Trains both models at once.
        loss_function = InverseFittingLoss()
        
        return elasticity_model, optimizer, loss_function
    
    def coordinates_to_input_tensor(coordinates:np.ndarray, device_to=DEVICE) -> torch.Tensor:
        ss_coordinates = preprocessing.StandardScaler()
        return torch.tensor(
            ss_coordinates.fit_transform(coordinates.reshape(-1, 2)),
            dtype=torch.float32,
            device=device_to)
    
    # Coordinate array should have dim = 1
    def disp_to_elas_input_tensor(coordinates:np.ndarray, 
                                device_to=DEVICE) -> torch.Tensor: # Uses input type
        ss_coordinates = preprocessing.StandardScaler()
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
        
        # Create image from coordinates
        coord_mat = torch.tensor(coordinates.reshape(DATA_SHAPE[0]+1, DATA_SHAPE[1]+1, 2),
                                    dtype=TENSOR_TYPE, device=DEVICE)
        
        coord_mat_x = coord_mat[:,:,0]
        result_x = coord_conv(coord_mat_x.reshape(1,1,DATA_SHAPE[0]+1,DATA_SHAPE[1]+1)).reshape(-1)
        
        coord_mat_y = coord_mat[:,:,1]
        result_y = coord_conv(coord_mat_y.reshape(1,1,DATA_SHAPE[0]+1,DATA_SHAPE[1]+1)).reshape(-1)
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
        
    def save_displacement_fit_eval(model:DisplacementFitModel, coordinate_tensor:torch.Tensor, 
                        file_name_ux, file_name_uy) ->None:
        model.eval()
        output = model(coordinate_tensor)
        # assert isinstance(output, torch.Tensor) # For intellisense typing convenience
        pred_ux = output[:, 0]
        pred_uy = output[:, 1]
        np.savetxt(file_name_ux, pred_ux.cpu().detach().numpy())
        np.savetxt(file_name_uy, pred_uy.cpu().detach().numpy())
    
    def initialize_and_train_model(
            elas_model:InverseModel,
            disp_model:DisplacementFitModel, 
            coordinates: np.ndarray, 
            displacement_data:np.ndarray,
            epochs_on_fitting = NUM_FITTING_ONLY_EPOCHS,
            device_to = DEVICE,
            epochs = NUM_EPOCHS,
            lr = LEARN_RATE,
            results_folder = OUTPUT_FOLDER,
            split_folders = SUB_FOLDERS) -> None:
        
        # TODO: try with a fixed displacement after fitting (consider both scenarios this and different loss terms
        # Since the boundary values seem to be off, could do loss without them (as they are inaccurate). 
        # Or more iterations of fitting.
        
        # Initialize elasticity model on correct device
        elas_model.to(device_to)
        elas_model.apply(InverseModel.init_weight_and_bias)
        
        # Initialize displacement model on correct device
        disp_model.to(device_to)
        assert isinstance(disp_model, DisplacementFitModel) # For intellisense typing convenience
        disp_model.apply(DisplacementFitModel.init_weight_and_bias)
        
        # optimizer, initialized depending on the current case
        # optimizer = torch.optim.Optimizer()
        
        loss_function = InverseFittingLoss()
        
        # Standardize the Discrete Coordinates, and fit to the coordinates (shape)
        # Of the output elasticity values used in loss calculations
        disp_coord = InverseFittingRunner.coordinates_to_input_tensor(
            coordinates, device_to=device_to
        )
        elas_coord = InverseFittingRunner.disp_to_elas_input_tensor(
            coordinates, device_to=device_to
        )
        
        displacement_data = torch.tensor(
            displacement_data, 
            dtype=TENSOR_TYPE,
            device=device_to)
        
        if STATE_MESSAGES: print("DEBUG: model initialized")
        if STATE_MESSAGES: print("DEBUG: starting displacement fitting")
        
        # Only Fitting
        optimizer = torch.optim.Adam(disp_model.parameters(), lr=lr)
        training_start_time=time.time()
        for e in range(epochs_on_fitting):
            print(f"Fitting Epoch {e} starting")
            disp_model.train()
            elas_model.train()
            epoch_start_time = time.time()
            for i in range (1, 1001):
                optimizer.zero_grad() # Resets the optimizer?
                output_disp = disp_model(disp_coord)
                pred_ux = output_disp[:, 0]
                pred_uy = output_disp[:, 1]
                loss = loss_function(
                    pred_ux=pred_ux,
                    pred_uy=pred_uy,
                    displacement_data=displacement_data,
                    only_fitting=True
                )
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                        print(f"Fitting Epoch: {e} [{i}/1000 ({i/10.0:.2f}%)]\tLoss: {loss.item():.6f}")
            e_time = time.time()-epoch_start_time
            print(f"Epoch{e} took {e_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(epochs_on_fitting-e) * e_time)}")
            
            if split_folders:
                InverseFittingRunner.save_displacement_fit_eval(
                    disp_model, disp_coord,
                    f"{results_folder}/pred_ux/fit{e}.txt",
                    f"{results_folder}/pred_uy/fit{e}.txt"
                )

            else:
                InverseFittingRunner.save_displacement_fit_eval(
                    disp_model, disp_coord,
                    f"{results_folder}/pred_ux_fit{e}.txt",
                    f"{results_folder}/pred_uy_fit{e}.txt"
                )
                
        # TODO: save the fitting model and reuse it for each time.
            
        
        if STATE_MESSAGES: print("DEBUG: training for elasticity constants")
        
        # Start Training
        optimizer = torch.optim.Adam(
            list(disp_model.parameters()) + list(elas_model.parameters()), 
            lr=lr)
        # TODO: depending on the the test, including or don't include disp_model.parameters (when not using loss_u)
        training_start_time=time.time()
        for e in range(epochs):
            print(f"Training Epoch {e} starting")
            disp_model.train()
            epoch_start_time = time.time()
            for i in range (1, 1001):
                optimizer.zero_grad() # Resets the optimizer?
                output_disp = disp_model(disp_coord)
                pred_ux = output_disp[:, 0]
                pred_uy = output_disp[:, 1]
                
                output_elas = elas_model(elas_coord)
                pred_E = output_elas[:, 0]
                pred_v = output_elas[:, 1]
                
                loss = loss_function(
                    pred_ux=pred_ux,
                    pred_uy=pred_uy,
                    pred_E=pred_E,
                    pred_v=pred_v,
                    displacement_data=displacement_data,
                    only_fitting=False
                )
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                        print(f"Training Epoch: {e} [{i}/1000 ({i/10.0:.2f}%)]\tLoss: {loss.item():.6f}")
            e_time = time.time()-epoch_start_time
            print(f"Epoch{e} took {e_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(epochs_on_fitting-e) * e_time)}")
            
            if split_folders:
                InverseFittingRunner.save_displacement_fit_eval(
                    disp_model, disp_coord,
                    f"{results_folder}/pred_ux/epoch{e}.txt",
                    f"{results_folder}/pred_uy/epoch{e}.txt"
                )
                InverseFittingRunner.save_elasticity_model_eval(
                    elas_model, elas_coord,
                    f"{results_folder}/pred_E/epoch{e}.txt",
                    f"{results_folder}/pred_v/epoch{e}.txt"
                )
            else:
                InverseFittingRunner.save_displacement_fit_eval(
                    disp_model, disp_coord,
                    f"{results_folder}/pred_ux_epoch{e}.txt",
                    f"{results_folder}/pred_uy_epoch{e}.txt"
                )
                InverseFittingRunner.save_elasticity_model_eval(
                    elas_model, elas_coord,
                    f"{results_folder}/pred_E_epoch{e}.txt",
                    f"{results_folder}/pred_v_epoch{e}.txt"
                )
        if STATE_MESSAGES: print("DEBUG: Training Finished")


def main() -> None:
    # LOAD Data. Called data_data sometimes as file is called _data.
    disp_coord_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/disp_coord')
    disp_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/disp_data')
    m_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/m_data')
    nu_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/nu_data')
    strain_coord_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/strain_coord')
    strain_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/strain_data')
    
    if STATE_MESSAGES: print("DEBUG: data imported")
    
    use_input = False
    epochs_on_fitting = 25
    epochs = 25
    if use_input:
        epochs_on_fitting = input("Epochs for fitting: ")
        epochs = input("Epochs for training")

    elas_model = InverseModel()
    disp_model = DisplacementFitModel()
    InverseFittingRunner.initialize_and_train_model(
        elas_model = elas_model,
        disp_model = disp_model,
        coordinates = disp_coord_data,
        displacement_data = disp_data_data,
        epochs = epochs, # Overriding default
        epochs_on_fitting = epochs_on_fitting, # Overriding default
    )
    
    # Save DNN
    type_string = "_d"
    torch.save(elas_model.state_dict(), 
        f"{OUTPUT_FOLDER}/inverse_model{type_string}_{TRIAL_NAME}_f{epochs_on_fitting}_e{epochs}.pt")
    
    if STATE_MESSAGES: print("DEBUG: Done")

if __name__ == "__main__":
    main()
