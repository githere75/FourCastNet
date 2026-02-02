import torch
import h5py
import numpy as np
from collections import OrderedDict
from networks.afnonet import AFNONet
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels

def load_custom_model(model, checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def run_full_year_benchmark(params_obj, weight_path, data_path, num_steps=1, stride=36):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with the correct depth for this specific checkpoint
    model = AFNONet(params_obj).to(device)
    model = load_custom_model(model, weight_path, device)

    # 1. Load Normalization Stats
    means = np.load("data/stats/global_means.npy")[0, :20].reshape(20, 1, 1)
    stds = np.load("data/stats/global_stds.npy")[0, :20].reshape(20, 1, 1)
    std_tensor = torch.as_tensor(stds).to(device).view(1, 20, 1, 1)
    
    # 2. Load Climatology for Anomaly Correlation Coefficient (ACC)
    # We subtract the time-mean to calculate true anomalies rather than raw correlation
    time_means_raw = np.load("data/stats/time_means.npy")[0, :20].reshape(20, 721, 1440)

    # 2. Slice it to 720 to match your model's img_size (720, 1440)
    time_means = time_means_raw[:, 0:720, :]    
    # 3. Create the tensor
    clim_std = torch.as_tensor((time_means - means) / stds).to(device).view(1, 20, 720, 1440).float()
    
    all_z500_rmse_m = [] # in Meters
    all_t850_rmse_k = [] # in Kelvin
    all_accs = []

    G = 9.81 # Gravity for Geopotential to Height conversion

    with h5py.File(data_path, 'r') as f:
        n_samples = f['fields'].shape[0]
        
        # stride=36 (9 days) is the official protocol for independent Z500 samples
        for start_idx in range(0, n_samples - num_steps, stride):
            
            raw_ic = f['fields'][start_idx : start_idx+1, :20, :720, :1440]
            current_input = torch.from_numpy((raw_ic - means) / stds).to(device).float()
            
            raw_targets = f['fields'][start_idx+1 : start_idx+num_steps+1, :20, :720, :1440]
            targets = torch.from_numpy((raw_targets - means) / stds).to(device).float()

            with torch.no_grad():
                for step in range(num_steps):
                    prediction = model(current_input)
                    
                    # SCALE TO PHYSICAL UNITS
                    phys_pred = prediction * std_tensor
                    phys_target = targets[step:step+1] * std_tensor
                    
                    # 3. Calculate Metrics
                    rmse_per_channel = weighted_rmse_torch_channels(phys_pred, phys_target)
                    
                    # ACC requires anomaly subtraction (Value - Climatology)
                    acc_per_channel = weighted_acc_torch_channels(prediction - clim_std, 
                                                                 targets[step:step+1] - clim_std)
                    
                    # Extract Key Benchmarks
                    # Convert Z500 Geopotential (m^2/s^2) to Height (m)
                    z500_rmse_meters = rmse_per_channel[0, 14].item() / G
                    all_z500_rmse_m.append(z500_rmse_meters)
                    
                    all_t850_rmse_k.append(rmse_per_channel[0, 5].item())
                    all_accs.append(acc_per_channel.mean().item())
                    
                    current_input = prediction 

            if start_idx % 144 == 0:
                print(f"Progress: {start_idx}/{n_samples} steps processed...")

    print(f"\n--- Final Benchmark Report (2018 Full Year) ---")
    print(f"Weight Path: {weight_path}")
    print(f"Avg Z500 RMSE: {np.mean(all_z500_rmse_m):.2f} m")
    print(f"Avg T850 RMSE: {np.mean(all_t850_rmse_k):.2f} K")
    print(f"Avg Mean ACC:  {np.mean(all_accs):.4f}")

if __name__ == "__main__":
    
    class ParamsObject:
        def __init__(self, depth, **entries):
            self.__dict__.update(entries)
            self.depth = depth
            self.in_channels, self.out_channels = 20, 20
            self.patch_size, self.num_blocks = 8, 8

    # Base configuration
    base_params = {
        'patch_size': 8, 'embed_dim': 768, 'num_blocks': 8,
        'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
    }

    # 1. Evaluate 11-layer Pruned Model (Dropped 1 layer)
    print("\nEvaluating base Model (Depth=12)...")
    pruned_params = ParamsObject(depth=12, **base_params)
    run_full_year_benchmark(pruned_params, "weights/backbone.ckpt", "data/out_of_sample/2018.h5")

    # # 2. Evaluate Fine-tuned Model (Depth=11)
    # print("\nEvaluating Fine-tuned Model (Depth=11)...")
    # ft_params = ParamsObject(depth=11, **base_params)
    # run_full_year_benchmark(ft_params, "weights/fine-tuned models/fine_tuned1_epoch_1.ckpt", "data/out_of_sample/2018.h5")