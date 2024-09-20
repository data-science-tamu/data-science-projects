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
# Code idea / how to from:
# https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1
# training function also based on / started from it.
class InverseLoss(torch.nn.Module):
    E_CONSTRAINT = 1
    def __init__(self, shape, mean_modulo = 1.0):
        super().__init__()
        self.shape = shape
        self.mean_modulo = mean_modulo

    def forward(self, pred_E, pred_v, strain):
        stress = self.calculate_stress(pred_E, pred_v, strain)
        loss_x, loss_y = self.calculate_loss_r(pred_E, stress)
        
        # Equation (13)
        loss_e = torch.abs(torch.mean(pred_E) - self.mean_modulo)
        
        return loss_x + loss_y + loss_e / 100

    # Based on equation (2), the elastic constitutive relation
    def calculate_stress(self, pred_E, pred_v, strain):
        # Strain comes stacked (assumption)
        E_stack = torch.stack([pred_E, pred_E, pred_E], dim=1)
        v_stack = torch.stack([pred_v, pred_v, pred_v], dim=1)        
        
        def c_matrix(v:float):
            return torch.tensor(
                np.array(
                    [ [1, v, 0], 
                      [v, 1, 0], 
                      [0, 0, (1-v)/2.0] ]))
            
        mat_res = torch.stack([ 
            torch.matmul(strain[i], c_matrix(v_stack[i, 0])) 
                for i in range(strain.shape[0]) 
        ])

        v2 = torch.square(v_stack)
        fraction = torch.divide(E_stack, 1 - v2)
        stress = torch.multiply(mat_res, fraction)
        return stress
        

    # Based on equation (8)
    def calculate_loss_r(self, pred_E, stress) -> tuple[float, float]:
        def conv2d(x, W):
            return torch.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'valid')

        # Young's Modulus, (calculating E_hat (9) for L_r equation (8))
        sum_kernel = np.array(
            [[[[1.0]], [[1.0]], [[1.0]]], 
            [[[1.0]], [[1.0]], [[1.0]]],
            [[[1.0]], [[1.0]], [[1.0]]], ]   
        )
        pred_E_matrix = torch.reshape(pred_E, [self.shape[0], self.shape[1]])
        pred_E_matrix_4d = torch.reshape(pred_E_matrix, [-1, self.shape[0], self.shape[1], 1])
        pred_E_conv = conv2d(pred_E_matrix_4d, sum_kernel)
        
        # Transform Stress to 4d (-1, 256, 256, -1)
        stress_xx = stress[:, 0]
        stress_yy = stress[:, 1]
        stress_xy = stress[:, 2]
        
        stress_xx_matrix = torch.reshape(stress_xx, [self.shape[0], self.shape[1]])
        stress_yy_matrix = torch.reshape(stress_yy, [self.shape[0], self.shape[1]])
        stress_xy_matrix = torch.reshape(stress_xy, [self.shape[0], self.shape[1]])
        # I don't know why the provided code reshapes it twice, but I won't change it
        stress_xx_matrix_4d = torch.reshape(stress_xx_matrix, [-1, self.shape[0], self.shape[1], 1])
        stress_yy_matrix_4d = torch.reshape(stress_yy_matrix, [-1, self.shape[0], self.shape[1], 1])
        stress_xy_matrix_4d = torch.reshape(stress_xy_matrix, [-1, self.shape[0], self.shape[1], 1])
        
        # Convolutions from paper - calculate derivatives of strain
        wx_conv_xx = np.array(
            [[[[-1.0]], [[-1.0]], [[-1.0]]], 
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[1.0]], [[1.0]], [[1.0]]], ])
        wx_conv_xy = np.array(
            [[[[1.0]], [[0.0]], [[-1.0]]], 
            [[[1.0]], [[0.0]], [[-1.0]]],
            [[[1.0]], [[0.0]], [[-1.0]]], ])
        wy_conv_yy = np.array(
            [[[[1.0]], [[0.0]], [[-1.0]]], 
            [[[1.0]], [[0.0]], [[-1.0]]],
            [[[1.0]], [[0.0]], [[-1.0]]], ])
        wy_conv_xy = np.array(
            [[[[-1.0]], [[-1.0]], [[-1.0]]], 
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[1.0]], [[1.0]], [[1.0]]], ])

        # Make tensors
        wx_conv_xx = torch.tensor(wx_conv_xx, dtype = torch.float32)
        wx_conv_xy = torch.tensor(wx_conv_xy, dtype = torch.float32)
        wy_conv_yy = torch.tensor(wy_conv_yy, dtype = torch.float32)
        wy_conv_xy = torch.tensor(wy_conv_xy, dtype = torch.float32)
        
        
        # From equilibrium condition
        fx_conv_xx = conv2d(stress_xx_matrix_4d, wx_conv_xx)
        fx_conv_xy = conv2d(stress_xy_matrix_4d, wx_conv_xy)
        fx_conv_sum = fx_conv_xx + fx_conv_xy # Result that should be 0

        fy_conv_yy = conv2d(stress_yy_matrix_4d, wy_conv_yy)
        fy_conv_xy = conv2d(stress_xy_matrix_4d, wy_conv_xy)
        fy_conv_sum = fy_conv_yy + fy_conv_xy # Result that should be 0

        # Normalization, doing equation (8), rest was calculating residual forces
        fx_conv_sum_norm = torch.divide(fx_conv_sum, pred_E_conv)
        fy_conv_sum_norm = torch.divide(fy_conv_sum, pred_E_conv)
        
        # Finally get value of loss
        loss_x = torch.mean(torch.abs(fx_conv_sum_norm))
        loss_y = torch.mean(torch.abs(fy_conv_sum_norm))
        return loss_x, loss_y


data_shape = [256, 256]
model = InverseModel()
model.apply(init_weight_and_bias)
optimizer = torch.optim.Adam(model.parameters(), lr=InverseModel.LEARN_RATE)
loss_function = InverseLoss(data_shape)

print(model)

def display_weights(model):
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            print("weights:", layer.state_dict()['weight'])
            print("bias:", layer.state_dict()['bias'])

quit()

# Training Sample Code
"""
# Code idea / how to from:
# https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
        output = network(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
"""


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


