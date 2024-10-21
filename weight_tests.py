# import the main code file
# TODO? Split each component of inverse_model_displacement into a file in
# its own folder?
import src.inverse_model_fitting as src
from src.inverse_model_fitting import torch
from src.inverse_model_fitting import np

from plot import animation as anim
from results.clear_folders import clear_folders

PATH_TO_DATA = "./data"
TRIAL_NAME = "m_z5_nu_z11"
OUTPUT_FOLDER = "./results"

# Modified Main Function to run
def test_weight(weight:float) -> None:
    # LOAD Data. Called data_data sometimes as file is called _data.
    disp_coord_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/disp_coord')
    disp_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/disp_data')
    m_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/m_data')
    nu_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/nu_data')
    strain_coord_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/strain_coord')
    strain_data_data = np.loadtxt(f'{PATH_TO_DATA}/compressible/{TRIAL_NAME}/strain_data')
    
    print("DEBUG: data imported")
        
    # Change weight to find best one
    src.InverseFittingLoss.WEIGHT_U = weight
    
    use_input = False
    epochs_on_fitting = 50
    epochs_on_training = 100
    if use_input:
        epochs_on_fitting = input("Epochs for fitting: ")
        epochs_on_training = input("Epochs for training")

    disp_model_name = f"{OUTPUT_FOLDER}/inverse_disp_model_{TRIAL_NAME}_f{epochs_on_fitting}_e{epochs_on_training}_u{src.InverseFittingLoss.WEIGHT_U}.pt"
    elas_model_name = f"{OUTPUT_FOLDER}/inverse_elas_model_{TRIAL_NAME}_f{epochs_on_fitting}_e{epochs_on_training}_u{src.InverseFittingLoss.WEIGHT_U}.pt"
    
    elas_model = src.InverseModel()
    disp_model = src.DisplacementFitModel()
    src.InverseFittingRunner.initialize_and_train_model(
        elas_model = elas_model,
        disp_model = disp_model,
        coordinate_array = disp_coord_data,
        displacement_array = disp_data_data,
        epochs = epochs_on_training, # Overriding default
        epochs_on_fitting = epochs_on_fitting, # Overriding default
    )
    
    # Save DNN
    torch.save(elas_model.state_dict(), elas_model_name)
    torch.save(disp_model.state_dict(), disp_model_name)
    print("DEBUG: Done")
    
    anim.save_both(epochs_on_fitting, epochs_on_training, TRIAL_NAME, f"wu{src.InverseFittingLoss.WEIGHT_U}")
    clear_folders(clear_fit=False)

def main():
    test_weight(1)
    test_weight(1.5)
    test_weight(0.5)
    test_weight(2)
    test_weight(0.1)
    test_weight(0.01)

if __name__ == "__main__":
    main()