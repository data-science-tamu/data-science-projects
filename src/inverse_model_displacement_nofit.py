import torch
import numpy as np

from sklearn import preprocessing

import time
from datetime import timedelta

from typing import Tuple

# Program setup
state_messages = True
default_cpu = False # If false defaults to gpu
tensor_type = torch.float32

# Selecting device to use. If use_cpu is true, will default
# to cpu, otherwise will use GPU if a GPU is available

if default_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if state_messages: print("DEBUG: torch device is", device)

# Training information, Default Values
learn_rate = 0.001
num_epochs = 200 # Each epoch is 1000 training "sessions"

# Output shape, assuming that displacement shape is 1 larger
# If data shape is (256, 256) then displacement is assumed at (257, 257)
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


class InverseLoss(torch.nn.Module):
    E_CONSTRAINT = 0.25
    
    def __init__(self, mean_modulo=E_CONSTRAINT):
        super(InverseLoss, self).__init__()
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
            dtype=tensor_type, device=device).reshape(1, 1, 2, 2))
        
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
            dtype=tensor_type, device=device).reshape(1, 1, 2, 2))
        
        # To manipulate the pred_E (9)
        self.sum_kernel = torch.tensor(
            [[1.0, 1.0, 1.0], 
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],], 
            dtype=tensor_type, device=device)
        
        # Used to calculate partials of equilibrium condition
        self.wx_conv_xx = torch.tensor(
            [[-1.0, -1.0, -1.0], 
             [0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0], ],
            dtype = tensor_type, device=device)
        self.wx_conv_xy = torch.tensor(
            [[1.0, 0.0, -1.0], 
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0], ],
            dtype = tensor_type, device=device)
        self.wy_conv_yy = torch.tensor(
            [[1.0, 0.0, -1.0], 
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0], ],
            dtype = tensor_type, device=device)
        self.wy_conv_xy = torch.tensor(
            [[-1.0, -1.0, -1.0], 
             [0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0], ],
            dtype = tensor_type, device=device)
    
    def forward(self, pred_E, pred_v, data, is_strain=True):
        if is_strain:
            strain = data
        else: # input is displacement
            strain = self.calculate_strain(displacement_stack=data)
        
        stress = self.calculate_stress(pred_E, pred_v, strain)
        loss_x, loss_y = self.calculate_loss_r(pred_E, stress)
        
        # Equation (13)
        loss_e = torch.abs(torch.mean(pred_E) - self.mean_modulo)
        
        return loss_x + loss_y + loss_e / 100.0
    
    # Probably better to call this once instead of upon every iteration
    # Seeing as it is static (in so much as it doesn't use the predicted
    # Elasticity values.)
    # This function takes an unprocessed displacement (raw data of shape [:, 2])
    # Assuming each entry is [u_x, u_y] ([u, v] from papers code)
    def calculate_strain(self, displacement_stack:torch.Tensor) -> torch.Tensor:

        ux_matrix = displacement_stack[:, 0].reshape(1, 1, data_shape[0]+1, data_shape[1]+1)
        uy_matrix = displacement_stack[:, 1].reshape(1, 1, data_shape[0]+1, data_shape[1]+1)

        e_xx = self.strain_conv_x(ux_matrix) # u_xx
        e_yy = self.strain_conv_y(uy_matrix) # u_yy
        r_xy = self.strain_conv_y(ux_matrix) + self.strain_conv_x(uy_matrix) # u_xy + u_yx
        
        # The following is from the paper, 
        # I don't yet know why it is multiplied by 100
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
            torch.ones(pred_v.shape, dtype=torch.float32, device=device),
            pred_v,
            torch.zeros(pred_v.shape, dtype=torch.float32, device=device), ##
            pred_v,
            torch.ones(pred_v.shape, dtype=torch.float32, device=device),
            torch.zeros(pred_v.shape, dtype=torch.float32, device=device), ##
            torch.zeros(pred_v.shape, dtype=torch.float32, device=device),
            torch.zeros(pred_v.shape, dtype=torch.float32, device=device),
            torch.divide(
                (torch.ones(pred_v.shape, dtype=torch.float32, device=device) - pred_v),
                torch.full(pred_v.shape, 2.0, dtype=torch.float32, device=device)
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
        
        pred_E_matrix = torch.reshape(pred_E, [data_shape[0], data_shape[1]])
        pred_E_matrix_4d = torch.reshape(pred_E_matrix, [-1, 1, data_shape[0], data_shape[1]])
        pred_E_conv = conv2d(pred_E_matrix_4d, self.sum_kernel)
        
        # Unstack the stress
        stress_xx = stress[:, 0]
        stress_yy = stress[:, 1]
        stress_xy = stress[:, 2]
        
        stress_xx_matrix_4d = torch.reshape(stress_xx, [-1, 1, data_shape[0], data_shape[1]])
        stress_yy_matrix_4d = torch.reshape(stress_yy, [-1, 1, data_shape[0], data_shape[1]])
        stress_xy_matrix_4d = torch.reshape(stress_xy, [-1, 1, data_shape[0], data_shape[1]])
        
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
    

class InverseRunner():
    def print_gpu_memory():
        if state_messages and device.type == 'cuda':
            print(torch.cuda.get_device_name(device=device))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(device=device)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(device=device)/1024**3,1), 'GB')
    
    def initialize_model(
                model:InverseModel, device_to=device, lr=learn_rate
            ) -> Tuple[InverseModel, torch.optim.Optimizer, InverseLoss]:
        model.to(device_to)
        model.apply(InverseModel.init_weight_and_bias)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_function = InverseLoss()
        
        return model, optimizer, loss_function
    
    # Coordinate array should have dim = 1
    def coordinates_to_input_tensors(coordinates:np.ndarray, 
                                input_type) -> torch.Tensor: # Uses input type
        ss_coordinates = preprocessing.StandardScaler()
        if input_type == InverseModel.IS_STRAIN:
            return torch.tensor(
                ss_coordinates.fit_transform(coordinates.reshape(-1, 2)),
                dtype=torch.float32,
                device=device)
        
        elif input_type == InverseModel.IS_DISPLACEMENT:
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
                [ 0.25,  0.25]], dtype=tensor_type, device=device).reshape(1, 1, 2, 2))
            
            # Create image from coordinates
            coord_mat = torch.tensor(coordinates.reshape(data_shape[0]+1, data_shape[1]+1, 2),
                                     dtype=tensor_type, device=device)
            
            coord_mat_x = coord_mat[:,:,0]
            result_x = coord_conv(coord_mat_x.reshape(1,1,data_shape[0]+1,data_shape[1]+1)).reshape(-1)
            
            coord_mat_y = coord_mat[:,:,1]
            result_y = coord_conv(coord_mat_y.reshape(1,1,data_shape[0]+1,data_shape[1]+1)).reshape(-1)
            strain_coordinates = torch.stack([result_x, result_y], dim=1).detach()
            
            return torch.tensor(
                ss_coordinates.fit_transform(strain_coordinates.cpu()),
                dtype=torch.float32,
                device=device)
    
    def save_model_eval(model:InverseModel, coordinate_tensor, file_name_E, file_name_v) -> None:
        model.eval()
        output = model(coordinate_tensor)
        assert isinstance(output, torch.Tensor) # For intellisense typing convenience
        pred_E = output[:, 0]
        pred_v = output[:, 1]
        np.savetxt(file_name_E, pred_E.cpu().detach().numpy())
        np.savetxt(file_name_v, pred_v.cpu().detach().numpy())
    
    def initialize_and_train_model(
            model:InverseModel, 
            coordinates: np.ndarray, 
            data:np.ndarray,
            data_type = InverseModel.IS_STRAIN,
            epochs = num_epochs,
            lr = learn_rate,
            save_final = False,
            results_folder = output_folder,
            split_folders = sub_folders) -> None:
        
        assert model.input_type == data_type
        
        model, optimizer, loss_function = InverseRunner.initialize_model(model, lr=lr)
        assert isinstance(model, InverseModel) # For intellisense typing convenience
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(loss_function, InverseLoss)
        
        # Standardize the Discrete Coordinates, and fit to the coordinates (shape)
        # Of the output elasticity values used in loss calculations
        coordinates = InverseRunner.coordinates_to_input_tensors(coordinates, input_type=data_type)
        
        # If data is displacement, pre-calculate strain as this calculation
        # isn't affected by the predicted Elasticity values
        data = torch.tensor(
            data, 
            dtype=torch.float32,
            device=device)
        if data_type == InverseModel.IS_DISPLACEMENT:
            strain = loss_function.calculate_strain(data).detach()
        elif data_type == InverseModel.IS_STRAIN:
            strain = data
        else:
            print("ERROR")
            return
        
        if state_messages: print("DEBUG: model initialized")
        
        # Start Training
        training_start_time=time.time()
        for e in range(epochs):
            print(f"Epoch {e} starting")
            model.train()
            epoch_start_time = time.time()
            for i in range (1, 1001):
                optimizer.zero_grad() # Reset the optimizer?
                output = model(coordinates)
                pred_E = output[:, 0]
                pred_v = output[:, 1]
                # calculate strain from predicted displacement
                # calc_pred_strain
                loss = loss_function(pred_E, pred_v, strain) # calc_pred_strain
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"Train Epoch: {e} [{i}/1000 ({i/10.0:.2f}%)]\tLoss: {loss.item():.6f}")
            
            e_time = time.time()-epoch_start_time
            print(f"Epoch{e} took {e_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs-e) * e_time)}")
            
            
            if split_folders:
                InverseRunner.save_model_eval(
                    model, coordinates,
                    f"{results_folder}/pred_E/epoch{e}.txt",
                    f"{results_folder}/pred_v/epoch{e}.txt"
                )

            else:
                InverseRunner.save_model_eval(
                    model, coordinates,
                    f"{results_folder}/pred_E_epoch{e}.txt",
                    f"{results_folder}/pred_v_epoch{e}.txt"
                )
        
        if state_messages: print("DEBUG: Training Finished")
        
        if save_final:
            if split_folders:
                InverseRunner.save_model_eval(
                    model, coordinates,
                    f"{results_folder}/pred_E/final.txt",
                    f"{results_folder}/pred_v/final.txt"
                )
            else:
                InverseRunner.save_model_eval(
                    model, coordinates,
                    f"{results_folder}/pred_E_final.txt",
                    f"{results_folder}/pred_v_final.txt"
                )


def main() -> None:
    # LOAD Data. Called data_data sometimes as file is called _data.
    disp_coord_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/disp_coord')
    disp_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/disp_data')
    m_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/m_data')
    nu_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/nu_data')
    strain_coord_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/strain_coord')
    strain_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/strain_data')
    
    if state_messages: print("DEBUG: data imported")
    
    epochs = 100
    # Not initialized yet. Can manually do so, or let train_model do all the work
    model = InverseModel(input_type=InverseModel.IS_DISPLACEMENT) 
    InverseRunner.initialize_and_train_model(
        model,
        coordinates = disp_coord_data,
        data = disp_data_data,
        data_type = InverseModel.IS_DISPLACEMENT,
        save_final = True,
        epochs = epochs, # Overriding default
    )
    
    # Save DNN
    if model.input_type == InverseModel.IS_DISPLACEMENT:
        type_string = "d"
    elif model.input_type == InverseModel.IS_STRAIN:
        type_string = "s" 
    torch.save(model.state_dict(), 
        f"{output_folder}/inverse_model_{type_string}_{trial_name}_e{epochs}.pt")
    
    if state_messages: print("DEBUG: Done")

if __name__ == "__main__":
    main()
