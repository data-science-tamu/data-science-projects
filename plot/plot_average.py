import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


num_epochs = 200 # Assumes 0 to num_epochs (not inclusive)

path_to_data = "./data"
trial_name = "m_z5_nu_z11"
m_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/m_data').reshape(256,256)
nu_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/nu_data').reshape(256,256)

path_E = "./results/pred_E/"
path_v = "./results/pred_v/"

fig, ((axE_exp, axE, axE_err), (axv_exp, axv, axv_err)) = plt.subplots(2, 3)

imE_exp = axE_exp.imshow(m_data)
imv_exp = axv_exp.imshow(nu_data)
colorbarE_exp = fig.colorbar(imE_exp, ax=axE_exp)
colorbarv_exp = fig.colorbar(imv_exp, ax=axv_exp)

final = "final"
pred_e = np.loadtxt(path_E + f"epoch{final}.txt").reshape(256,256)
pred_v = np.loadtxt(path_v + f"epoch{final}.txt").reshape(256,256)


final = "final"
total_pred_E = np.loadtxt(path_E + f"epoch{final}.txt").reshape(256,256)
total_pred_v = np.loadtxt(path_v + f"epoch{final}.txt").reshape(256,256)
n = 1
for i in range(num_epochs-50, num_epochs):
    total_pred_E += np.loadtxt(path_E + f"epoch{i}.txt").reshape(256,256)
    total_pred_v += np.loadtxt(path_v + f"epoch{i}.txt").reshape(256,256)
    n += 1
total_pred_E /= n
total_pred_E /= 4

total_pred_v /= n

imE = axE.imshow(total_pred_E)
imv = axv.imshow(total_pred_v)

colorbarE_pred = fig.colorbar(imE, ax=axE)
colorbarE_pred = fig.colorbar(imv, ax=axv)

E_err = np.abs((total_pred_E-m_data)/m_data)
imE_err = axE_err.imshow(E_err)
v_err = np.abs((total_pred_v-nu_data)/nu_data)
imv_err = axv_err.imshow(v_err)

colorbarE_pred = fig.colorbar(imE_err, ax=axE_err)
colorbarE_pred = fig.colorbar(imv_err, ax=axv_err)

print(E_err.sum())
print(v_err.sum())
plt.show()