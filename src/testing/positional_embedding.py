import torch
from torch import nn
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


# p.add_argument('--model_type', type=str, default='sine',
#               help='Options currently are "sine" (all sine activations), 
#           "relu" (all relu activations,'
#           '"nerf" (relu activations and positional encoding as in NeRF), 
#           "rbf" (input rbf layer, rest relu),'
#           'and in the future: "mixed" (first layer sine, other layers tanh)')

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
    
    
# Testing positional Encoding
strain_data = np.loadtxt("./strain_data").reshape(256,256,3)
# plt.imshow(strain_data)

strain_coord = np.loadtxt("./strain_coord").reshape(256,256,2)
# plt.imshow(strain_coord[:,:,0])

strain_coord = torch.Tensor(strain_coord)

pos_encode = PosEncodingNeRF(2, sidelength=[256,256])
thing = pos_encode(strain_coord)
thing = thing.squeeze().reshape([-1,pos_encode.out_dim])
print(thing[:,9::2], thing.size())
plt.imshow(thing[:,7:11:2], interpolation='nearest', aspect='auto', cmap='viridis')
plt.colorbar()
plt.show()
# for i in range(1000):
#     plt.plot(thing[6*i])
plt.show()