import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


num_epochs = 100 # Assumes 0 to num_epochs (not inclusive)

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

img = []
final = "final"
pred_e = np.loadtxt(path_E + f"epoch{final}.txt").reshape(256,256)
pred_v = np.loadtxt(path_v + f"epoch{final}.txt").reshape(256,256)
imE = axE.imshow(pred_e)
imv = axv.imshow(pred_v)
imE_err = axE_err.imshow(np.abs((pred_e-m_data)/m_data))
imv_err = axv_err.imshow(np.abs((pred_v-nu_data)/nu_data))

colorbarE_pred = fig.colorbar(imE, ax=axE)
colorbarE_pred = fig.colorbar(imv, ax=axv)

img.append([imE, imv, imE_err, imv_err])

for i in range(0, num_epochs):
    pred_e = np.loadtxt(path_E + f"epoch{i}.txt").reshape(256,256)
    pred_v = np.loadtxt(path_v + f"epoch{i}.txt").reshape(256,256)
    imE = axE.imshow(pred_e)
    imv = axv.imshow(pred_v)
    imE_err = axE_err.imshow(np.abs((pred_e-m_data)/m_data))
    imv_err = axv_err.imshow(np.abs((pred_v-nu_data)/nu_data))
    img.append([imE, imv, imE_err, imv_err])
    colorbarE_pred.update_normal(imE)
    colorbarE_pred.update_normal(imv)
    
# imE = axE.imshow(np.loadtxt(path_E + f"epochfinal.txt").reshape(256,256))
# imv = axv.imshow(np.loadtxt(path_v + f"epochfinal.txt").reshape(256,256))
# img.append([imE, imv])


ani = animation.ArtistAnimation(fig, img, interval=400, blit=True, repeat_delay=500)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
# wm = plt.get_current_fig_manager()
# wm.window.state('zoomed')
plt.show()