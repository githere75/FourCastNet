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

def run_full_year_benchmark(params_obj, weight_path, data_path, num_steps=1, stride=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFNONet(params_obj).to(device)
    model = load_custom_model(model, weight_path, device)

    # 1. Load Normalization Stats
    means = np.load("data/stats/global_means.npy")[0, :20].reshape(20, 1, 1)
    stds = np.load("data/stats/global_stds.npy")[0, :20].reshape(20, 1, 1)
    std_tensor = torch.as_tensor(stds).to(device).view(1, 20, 1, 1)

    all_z500_rmse = []
    all_t850_rmse = []
    all_accs = []
    

    with h5py.File(data_path, 'r') as f:
        n_samples = f['fields'].shape[0]
        # Use a stride to evaluate the whole year faster (e.g., every 48 hours)
        for start_idx in range(0, n_samples - num_steps, stride):
            
            raw_ic = f['fields'][start_idx : start_idx+1, :20, :720, :1440]
            current_input = torch.from_numpy((raw_ic - means) / stds).to(device).float()
            
            raw_targets = f['fields'][start_idx+1 : start_idx+num_steps+1, :20, :720, :1440]
            targets = torch.from_numpy((raw_targets - means) / stds).to(device).float()

            with torch.no_grad():
                for step in range(num_steps):
                    prediction = model(current_input)
                    
                    # SCALE TO PHYSICAL UNITS FOR BENCHMARK COMPARISON
                    phys_pred = prediction * std_tensor
                    phys_target = targets[step:step+1] * std_tensor
                    
                    # Calculate RMSE per channel [1, 20]
                    rmse_per_channel = weighted_rmse_torch_channels(phys_pred, phys_target)
                    acc_per_channel = weighted_acc_torch_channels(prediction, targets[step:step+1])
                    
                    # Extract Key Benchmarks: Z500 (Idx 14) and T850 (Idx 5)
                    all_z500_rmse.append(rmse_per_channel[0, 14].item())
                    all_t850_rmse.append(rmse_per_channel[0, 5].item())
                    all_accs.append(acc_per_channel.mean().item())
                    
                    current_input = prediction 

            if start_idx % 100 == 0:
                print(f"Progress: {start_idx}/{n_samples} steps processed...")

    # Report results in units comparable to literature
    print(f"\n--- Final Benchmark Report (2018 Full Year) ---")
    print(f"Weight Path: {weight_path}")
    print(f"Avg Z500 RMSE: {np.mean(all_z500_rmse):.2f} m^2/s^2")
    print(f"Avg T850 RMSE: {np.mean(all_t850_rmse):.2f} K")
    print(f"Avg Mean ACC:  {np.mean(all_accs):.4f}")

if __name__ == "__main__":
    # Ensure depth matches the specific model you are testing!
    # Original=12, your pruned versions=6 or 9.
    params = {
        'patch_size': 8, 'embed_dim': 768, 'depth': 12, 'num_blocks': 8,
        'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
    }

    class ParamsObject:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            self.in_channels, self.out_channels = 20, 20
            self.patch_size, self.num_blocks = 8, 8

    # Test the backbone first to establish a baseline
    params_obj = ParamsObject(**params)
    run_full_year_benchmark(params_obj, "weights/backbone.ckpt", "data/out_of_sample/2018.h5", num_steps=1)

