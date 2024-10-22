import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

FITTING_NAME = "fit"
EPOCH_NAME = "epoch"

def save_elasticity(
        # Must specify
        num_epochs:int,
        trial_name:str,
        output_tag :str,
        
        # Optional settings
        path_to_data = "./data",
        path_E = "./results/pred_E/",
        path_v = "./results/pred_v/",
    ):

    path_to_data = "./data"
    trial_name = "m_z5_nu_z11"
    m_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/m_data').reshape(256,256)
    nu_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/nu_data').reshape(256,256)


    fig, ((axE_exp, axE, axE_err), (axv_exp, axv, axv_err)) = plt.subplots(2, 3)

    imE_exp = axE_exp.imshow(m_data)
    imv_exp = axv_exp.imshow(nu_data)
    colorbarE_exp = fig.colorbar(imE_exp, ax=axE_exp)
    colorbarv_exp = fig.colorbar(imv_exp, ax=axv_exp)

    img = []
    final = f"epoch{num_epochs-1}"
    pred_e = np.loadtxt(path_E + f"{final}.txt").reshape(256,256)
    pred_v = np.loadtxt(path_v + f"{final}.txt").reshape(256,256)
    imE = axE.imshow(pred_e)
    imv = axv.imshow(pred_v)
    imE_err = axE_err.imshow(np.abs(pred_e-m_data), cmap='turbo')
    imv_err = axv_err.imshow(np.abs(pred_v-nu_data), cmap='turbo')

    print(np.abs(pred_e-m_data).sum())
    print(np.abs(pred_v-nu_data).sum())

    colorbarE_pred = fig.colorbar(imE, ax=axE)
    colorbarv_pred = fig.colorbar(imv, ax=axv)
    colorbarE_err = fig.colorbar(imE_err, ax=axE_err)
    colorbarv_err = fig.colorbar(imv_err, ax=axv_err)

    # img.append([imE, imv, imE_err, imv_err])


    for i in range(0, num_epochs):
        pred_e = np.loadtxt(path_E + f"epoch{i}.txt").reshape(256,256)
        pred_v = np.loadtxt(path_v + f"epoch{i}.txt").reshape(256,256)
        imE = axE.imshow(pred_e)
        imv = axv.imshow(pred_v)
        imE_err = axE_err.imshow(np.abs(pred_e-m_data), cmap='turbo')
        imv_err = axv_err.imshow(np.abs(pred_v-nu_data), cmap='turbo')
        img.append([imE, imv, imE_err, imv_err])
        colorbarE_pred.update_normal(imE)
        colorbarv_pred.update_normal(imv)
        colorbarE_err.update_normal(imE_err)
        colorbarv_err.update_normal(imv_err)
        
    fig.set_size_inches(16, 9)
    anim = animation.ArtistAnimation(fig, img, interval=400, blit=True, repeat_delay=500)

    if output_tag:
        output_tag = "_" + output_tag
    f = f"./results/{trial_name}{output_tag}_{num_epochs}e_elas.gif"
    writer_gif = animation.PillowWriter(fps=5) 
    anim.save(f, writer=writer_gif)
    
    
def save_displacement(
        # Must specify
        num_fits:int, # Assumes 0 to num_epochs (not inclusive)
        num_epochs:int,
        trial_name:str,
        output_tag :str,
        
        # Optional settings
        save_fit = True,
        save_epoch = True,
        path_to_data = "./data",
        disp_dimension = 257,
        path_ux = "./results/pred_ux/",
        path_uy = "./results/pred_uy/",
    ):
    
    disp_data = np.loadtxt(f'{path_to_data}/compressible/{trial_name}/disp_data')

    ux_data = disp_data[:, 0].reshape(disp_dimension, disp_dimension)
    uy_data = disp_data[:, 1].reshape(disp_dimension, disp_dimension)


    fig, ((axx_exp, axx, axx_err), (axy_exp, axy, axy_err)) = plt.subplots(2, 3)

    imx_exp = axx_exp.imshow(ux_data)
    imy_exp = axy_exp.imshow(uy_data)
    colorbarx_exp = fig.colorbar(imx_exp, ax=axx_exp)
    colorbary_exp = fig.colorbar(imy_exp, ax=axy_exp)

    img = []
    final = f"fit{num_fits-1}"
    pred_ux = np.loadtxt(path_ux + f"{final}.txt").reshape(disp_dimension,disp_dimension)
    pred_uy = np.loadtxt(path_uy + f"{final}.txt").reshape(disp_dimension,disp_dimension)
    imE = axx.imshow(pred_ux)
    imv = axy.imshow(pred_uy)
    imx_err = axx_err.imshow(np.abs(pred_ux-ux_data), cmap='turbo')
    imy_err = axy_err.imshow(np.abs(pred_uy-uy_data), cmap='turbo')

    print(np.abs(pred_ux-ux_data).sum())
    print(np.abs(pred_uy-uy_data).sum())

    colorbarx_pred = fig.colorbar(imE, ax=axx)
    colorbary_pred = fig.colorbar(imv, ax=axy)
    colorbarx_err = fig.colorbar(imx_err, ax=axx_err)
    colorbary_err = fig.colorbar(imy_err, ax=axy_err)

    # img.append([imE, imv, imx_err, imy_err])

    if save_fit:
        for i in range(0, num_fits):
            pred_ux = np.loadtxt(path_ux + f"{FITTING_NAME}{i}.txt").reshape(disp_dimension,disp_dimension)
            pred_uy = np.loadtxt(path_uy + f"{FITTING_NAME}{i}.txt").reshape(disp_dimension,disp_dimension)
            imE = axx.imshow(pred_ux)
            imv = axy.imshow(pred_uy)
            imx_err = axx_err.imshow(np.abs(pred_ux-ux_data), cmap='turbo')
            imy_err = axy_err.imshow(np.abs(pred_uy-uy_data), cmap='turbo')
            img.append([imE, imv, imx_err, imy_err])
            colorbarx_pred.update_normal(imE)
            colorbary_pred.update_normal(imv)
            colorbarx_err.update_normal(imx_err)
            colorbary_err.update_normal(imy_err)

    if save_epoch:
        for i in range(0, num_epochs):
            pred_ux = np.loadtxt(path_ux + f"{EPOCH_NAME}{i}.txt").reshape(disp_dimension,disp_dimension)
            pred_uy = np.loadtxt(path_uy + f"{EPOCH_NAME}{i}.txt").reshape(disp_dimension,disp_dimension)
            imE = axx.imshow(pred_ux)
            imv = axy.imshow(pred_uy)
            imx_err = axx_err.imshow(np.abs(pred_ux-ux_data), cmap='turbo')
            imy_err = axy_err.imshow(np.abs(pred_uy-uy_data), cmap='turbo')
            img.append([imE, imv, imx_err, imy_err])
            colorbarx_pred.update_normal(imE)
            colorbary_pred.update_normal(imv)
            colorbarx_err.update_normal(imx_err)
            colorbary_err.update_normal(imy_err)
        

    fig.set_size_inches(16, 9)
    anim = animation.ArtistAnimation(fig, img, interval=400, blit=True, repeat_delay=500)

    training_tag = (f"_{num_epochs}e" if save_epoch else "") + (f"_{num_fits}f" if save_fit else "")
    if output_tag:
        output_tag = "_" + output_tag
    f = f"./results/{trial_name}{output_tag}{training_tag}_disp.gif" 
    writer_gif = animation.PillowWriter(fps=5) 
    anim.save(f, writer=writer_gif)
 
def save_both(
        num_fits:int, # Assumes 0 to num_epochs (not inclusive)
        num_epochs:int,
        trial_name:str,
        output_tag :str,
    ):
    # save_displacement(num_fits, num_epochs, trial_name, output_tag, save_epoch=False)
    # save_displacement(num_fits, num_epochs, trial_name, output_tag, save_fit=False)
    save_elasticity(num_epochs, trial_name, output_tag)
   
   


 
if __name__ == "__main__":
    save_both(50, 100, "m_z5_nu_z11", f"_1wd")
    pass