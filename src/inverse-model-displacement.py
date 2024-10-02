import torch
import numpy as np

from sklearn import preprocessing

import time
from datetime import timedelta

# Program setup
state_messages = True
use_cpu = False

# Selecting device to use. If use_cpu is true, will default
# to cpu, otherwise will use GPU if a GPU is available

if use_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if state_messages: print("DEBUG: torch device is", device, "\n")

if state_messages: print("DEBUG: Packages imported, ")

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
    # Default Model Configuration.
    # Changing the input and output sizes would require modifying the
    # loss function
    INPUT_SIZE = 2
    OUTPUT_SIZE = 2 # [0] is pred_E, [1] is pred_v
    NUM_NEURON = 128
    NUM_HIDDEN_LAYERS = 16
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
        
    def forward(self, x): # TODO: make this take displacements instead of strain
        # [u_x, u_y] for each row
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")( getattr(self, f"hidden{i}")(x) )
        x = self.act_out(self.out(x))
        return x


# Function for initializing weights and biases. Modifying this will
# only modify the initial values.
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

   
# For view model information (not really used)
def display_weights(model:torch.nn.Module):
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            print("weights:", layer.state_dict()['weight'])
            print("bias:", layer.state_dict()['bias'])


# Calculating Loss
class InverseLoss(torch.nn.Module):
    E_CONSTRAINT = 1.0
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
        
    def forward(self, pred_E, pred_v, displacement):
        strain = self.calculate_strain(displacement) # TODO:
        stress = self.calculate_stress(pred_E, pred_v, strain)
        loss_x, loss_y = self.calculate_loss_r(pred_E, stress)
        
        # Equation (13)
        loss_e = torch.abs(torch.mean(pred_E) - self.mean_modulo)
        
        return loss_x + loss_y + loss_e / 100.0
    
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
        

def print_gpu_memory():
    if state_messages and device.type == 'cuda':
        print(torch.cuda.get_device_name(device=device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(device=device)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(device=device)/1024**3,1), 'GB')


def main():
    # import data
    # Import Training Data
    # disp_coord_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/disp_coord')
    # disp_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/disp_data')
    # m_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/m_data')
    # nu_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/nu_data')
    strain_coord_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/strain_coord')
    strain_data_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/strain_data')

    # Standardize the inputs
    # Reshape to guarantee the correct size. (-1 indicates any size)
    ss_coordinates = preprocessing.StandardScaler()
    strain_coord = torch.tensor(
        ss_coordinates.fit_transform(strain_coord_data.reshape(-1, 2)),
        dtype=torch.float32,
        device=device)
    strain_data = torch.tensor(
        strain_data_data, 
        dtype=torch.float32,
        device=device)
    
    if state_messages: print("DEBUG: data imported")
    
    model = InverseModel()
    model.to(device)
    model.apply(init_weight_and_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_function = InverseLoss()
    
    if state_messages: print("DEBUG: model initialized")
        
    # train model
    # Train the model (does one epoch)
    training_start_time = 0
    def train(epoch, coordinate_data, strain_data):
        model.train()
        epoch_start_time = time.time()
        for i in range (1, 1001):
            optimizer.zero_grad() # Reset the optimizer?
            output = model(coordinate_data)
            pred_E = output[:, 0]
            pred_v = output[:, 1]
            loss = loss_function(pred_E, pred_v, strain_data)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Train Epoch: {epoch} [{i}/1000 ({i/10.0:.2f}%)]\tLoss: {loss.item():.6f}")
        
        e_time = time.time()-epoch_start_time
        print(f"Epoch{e} took {e_time} seconds.")
        print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
        print(f"Estimated time remaining is {timedelta(seconds=(num_epochs-e) * e_time)}")
        
    def save_current_state(e):
        model.eval()
        output = model(strain_coord)
        pred_E = output[:, 0]
        pred_v = output[:, 1]
        
        print(f"Saving Epoch {e}")
        if sub_folders:
            np.savetxt(f"{output_folder}/pred_E/epoch{e}.txt", pred_E.cpu().detach().numpy())
            np.savetxt(f"{output_folder}/pred_v/epoch{e}.txt", pred_v.cpu().detach().numpy())
        else:
            np.savetxt(f"{output_folder}/pred_E_epoch{e}.txt", pred_E.cpu().detach().numpy())
            np.savetxt(f"{output_folder}/pred_v_epoch{e}.txt", pred_v.cpu().detach().numpy())
    
    # print(disp_coord)
    training_start_time=time.time()
    for e in range(num_epochs):
        print(f"epoch {e} starting")
        train(e, strain_coord, strain_data)
        save_current_state(e)

    if state_messages: print("DEBUG: Training Finished")
    save_current_state("final")
    if state_messages: print("DEBUG: Done")



if __name__ == "__main__":
    main()