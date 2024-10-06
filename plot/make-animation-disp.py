import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


num_fits = 25 # Assumes 0 to num_epochs (not inclusive)
num_epochs = 21
fitting_name = "fit"
epoch_name = "epoch"
output_tag = ""

path_to_data = "./data"
trial_name = "m_z5_nu_z11"
dimension = 257
disp_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/disp_data')

ux_data = disp_data[:, 0].reshape(dimension, dimension)
uy_data = disp_data[:, 1].reshape(dimension, dimension)

path_ux = "./results/pred_ux/"
path_uy = "./results/pred_uy/"

fig, ((axx_exp, axx, axx_err), (axy_exp, axy, axy_err)) = plt.subplots(2, 3)

imx_exp = axx_exp.imshow(ux_data)
imy_exp = axy_exp.imshow(uy_data)
colorbarx_exp = fig.colorbar(imx_exp, ax=axx_exp)
colorbary_exp = fig.colorbar(imy_exp, ax=axy_exp)

img = []
final = "fit24"
pred_ux = np.loadtxt(path_ux + f"{final}.txt").reshape(dimension,dimension)
pred_uy = np.loadtxt(path_uy + f"{final}.txt").reshape(dimension,dimension)
imE = axx.imshow(pred_ux)
imv = axy.imshow(pred_uy)
imx_err = axx_err.imshow(np.abs((pred_ux-ux_data)/ux_data))
imy_err = axy_err.imshow(np.abs((pred_uy-uy_data)/uy_data))

print(np.abs((pred_ux-ux_data)/ux_data).sum())
print(np.abs((pred_uy-uy_data)/uy_data).sum())

colorbarx_pred = fig.colorbar(imE, ax=axx)
colorbary_pred = fig.colorbar(imv, ax=axy)

img.append([imE, imv, imx_err, imy_err])


for i in range(0, num_fits):
    pred_ux = np.loadtxt(path_ux + f"{fitting_name}{i}.txt").reshape(dimension,dimension)
    pred_uy = np.loadtxt(path_uy + f"{fitting_name}{i}.txt").reshape(dimension,dimension)
    imE = axx.imshow(pred_ux)
    imv = axy.imshow(pred_uy)
    imx_err = axx_err.imshow(np.abs((pred_ux-ux_data)/ux_data))
    imy_err = axy_err.imshow(np.abs((pred_uy-uy_data)/uy_data))
    img.append([imE, imv, imx_err, imy_err])
    colorbarx_pred.update_normal(imE)
    colorbary_pred.update_normal(imv)

for i in range(0, num_epochs):
    pred_ux = np.loadtxt(path_ux + f"{epoch_name}{i}.txt").reshape(dimension,dimension)
    pred_uy = np.loadtxt(path_uy + f"{epoch_name}{i}.txt").reshape(dimension,dimension)
    imE = axx.imshow(pred_ux)
    imv = axy.imshow(pred_uy)
    imx_err = axx_err.imshow(np.abs((pred_ux-ux_data)/ux_data))
    imy_err = axy_err.imshow(np.abs((pred_uy-uy_data)/uy_data))
    img.append([imE, imv, imx_err, imy_err])
    colorbarx_pred.update_normal(imE)
    colorbary_pred.update_normal(imv)
    

# imE = axx.imshow(np.loadtxt(path_ux + f"epochfinal.txt").reshape(dimension,dimension))
# imv = axy.imshow(np.loadtxt(path_uy + f"epochfinal.txt").reshape(dimension,dimension))
# img.append([imE, imv])

fig.set_size_inches(16, 9)
anim = animation.ArtistAnimation(fig, img, interval=400, blit=True, repeat_delay=500)

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

# f = f"./results/{trial_name}{output_tag}_{num_epochs}e.gif" 
# writer_gif = animation.PillowWriter(fps=5) 
# anim.save(f, writer=writer_gif)

plt.show()