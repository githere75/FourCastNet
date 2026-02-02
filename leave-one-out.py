import torch
import h5py
import numpy as np
from collections import OrderedDict
from networks.afnonet import AFNONet
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels

# Variable names for the final benchmark report
VAR_NAMES = [
    "u10", "v10", "t2m", "sp", "mslp", "t850", "u1000", "v1000", "z1000",
    "u850", "v850", "z850", "u500", "z50", "z500", "v500", "t500", "r500", "r850", "tcwv"
]

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

def run_sensitivity_benchmark(params_obj, weight_path, data_path, channel_to_mask, stride=36):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFNONet(params_obj).to(device)
    model = load_custom_model(model, weight_path, device)

    # 1. Load Normalization Stats
    means = np.load("data/stats/global_means.npy")[0, :20].reshape(20, 1, 1)
    stds = np.load("data/stats/global_stds.npy")[0, :20].reshape(20, 1, 1)
    std_tensor = torch.as_tensor(stds).to(device).view(1, 20, 1, 1)
    
    # 2. Load Climatology for True ACC calculation
    time_means_raw = np.load("data/stats/time_means.npy")[0, :20].reshape(20, 721, 1440)
    time_means = time_means_raw[:, 0:720, :] 
    # Standardize the climatology to match model space
    clim_std = torch.as_tensor((time_means - means) / stds).to(device).view(1, 20, 720, 1440).float()

    all_z500_rmse = []
    all_z500_acc = []

    with h5py.File(data_path, 'r') as f:
        n_samples = f['fields'].shape[0]
        # Using a stride of 9 days (36 steps) for Z500 independence
        for start_idx in range(0, n_samples - 1, stride):
            
            raw_ic = f['fields'][start_idx : start_idx+1, :20, :720, :1440]
            
            # --- STRATEGY: MASK INPUT PART ---
            # Simulate "dropping" the channel by feeding it the global mean
            raw_ic[:, channel_to_mask, :, :] = means[channel_to_mask, 0, 0]
            
            current_input = torch.from_numpy((raw_ic - means) / stds).to(device).float()
            target_raw = f['fields'][start_idx+1 : start_idx+2, :20, :720, :1440]
            target_norm = torch.from_numpy((target_raw - means) / stds).to(device).float()

            with torch.no_grad():
                prediction = model(current_input)
                
                # RMSE in m^2/s^2 for Z500
                phys_pred = prediction * std_tensor
                phys_target = target_norm * std_tensor
                rmse_per_channel = weighted_rmse_torch_channels(phys_pred, phys_target)
                
                # ACC: subtract climatology from prediction and target
                acc_per_channel = weighted_acc_torch_channels(prediction - clim_std, target_norm - clim_std)
                
                # Z500 is Index 14
                all_z500_rmse.append(rmse_per_channel[0, 14].item())
                all_z500_acc.append(acc_per_channel[0, 14].item())

    return np.mean(all_z500_rmse), np.mean(all_z500_acc)

if __name__ == "__main__":
    params = {
        'patch_size': 8, 'embed_dim': 768, 'depth': 12, 'num_blocks': 8,
        'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
    }

    class ParamsObject:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            self.in_channels, self.out_channels = 20, 20
            self.patch_size, self.num_blocks = 8, 8

    params_obj = ParamsObject(**params)
    weight_file = "weights/backbone.ckpt"
    data_file = "data/out_of_sample/2018.h5"

    print(f"Starting Multi-Metric Sensitivity Analysis...")
    results = []

    for i in range(20):
        rmse, acc = run_sensitivity_benchmark(params_obj, weight_file, data_file, channel_to_mask=i)
        results.append({"var": VAR_NAMES[i], "rmse": rmse, "acc": acc})
        print(f"Masked {VAR_NAMES[i]:<8} | RMSE: {rmse:.2f} m^2/s^2 | ACC: {acc:.4f}")

    # Sorting by lowest RMSE to find the least impactful variables
    sorted_results = sorted(results, key=lambda x: x['rmse'])
    
    print("\n--- Top Candidates to Drop (Lowest RMSE Impact) ---")
    for item in sorted_results[:5]:
        print(f"Candidate: {item['var']:<8} | RMSE: {item['rmse']:.2f} | ACC: {item['acc']:.4f}")