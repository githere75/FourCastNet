import torch
import h5py
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from networks.afnonet import AFNONet

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = 9.81                     # gravity (only needed if you later convert Z → height)
N_CHANNELS = 20
Z500_IDX = 14
T850_IDX = 5

# =========================================================
# MODEL LOADING
# =========================================================
def load_custom_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)

    clean_state = OrderedDict()
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


# =========================================================
# LATITUDE WEIGHTS (exact notebook definition)
# =========================================================
def latitude_weights(n_lat, device):
    lats = torch.linspace(90, -90, n_lat, device=device)
    weights = torch.cos(torch.deg2rad(lats))
    weights = weights / weights.mean()
    return weights  # [H]


# =========================================================
# WEIGHTED RMSE (per channel, per timestep)
# =========================================================
def weighted_rmse_channels(pred, target):
    """
    pred, target: [B, C, H, W] in physical units
    returns: RMSE per channel [C]
    """
    _, C, H, W = pred.shape
    lat_w = latitude_weights(H, pred.device).view(1, 1, H, 1)

    se = (pred - target) ** 2
    mse = (se * lat_w).mean(dim=(0, 2, 3))
    return torch.sqrt(mse)


# =========================================================
# WEIGHTED ACC (per channel, per timestep)
# =========================================================
def weighted_acc_channels(pred, target):
    """
    pred, target: [B, C, H, W]
    ACC computed on anomalies (no climatology file)
    """
    _, C, H, W = pred.shape
    lat_w = latitude_weights(H, pred.device).view(1, 1, H, 1)

    pred = pred - pred.mean(dim=(2, 3), keepdim=True)
    target = target - target.mean(dim=(2, 3), keepdim=True)

    num = (pred * target * lat_w).sum(dim=(0, 2, 3))
    den = torch.sqrt(
        (pred**2 * lat_w).sum(dim=(0, 2, 3)) *
        (target**2 * lat_w).sum(dim=(0, 2, 3))
    )

    return num / den


# =========================================================
# MAIN EVALUATION
# =========================================================
def run_evaluation(
    params_obj,
    weight_path,
    data_path,
    num_steps=4,
    stride=8
):
    model = AFNONet(params_obj).to(DEVICE)
    model = load_custom_model(model, weight_path)

    # Normalization stats (DO NOT CHANGE)
    means = np.load("data/stats/global_means.npy")[0, :N_CHANNELS].reshape(1, N_CHANNELS, 1, 1)
    stds  = np.load("data/stats/global_stds.npy")[0, :N_CHANNELS].reshape(1, N_CHANNELS, 1, 1)

    means = torch.tensor(means, device=DEVICE)
    stds  = torch.tensor(stds, device=DEVICE)

    z500_rmse_all = []
    t850_rmse_all = []
    acc_all = []

    with h5py.File(data_path, "r") as f:
        n_samples = f["fields"].shape[0]

        for t in tqdm(
            range(0, n_samples - num_steps, stride),
            desc="Evaluating FourCastNet (2018)",
            unit="step"
        ):
            # Initial condition
            x0 = f["fields"][t:t+1, :N_CHANNELS, :720, :]
            x = torch.tensor((x0 - means.cpu().numpy()) / stds.cpu().numpy(),
                             device=DEVICE, dtype=torch.float32)

            for step in range(num_steps):
                with torch.no_grad():
                    x = model(x)

                # De-normalize
                phys_pred = x * stds + means
                phys_tgt = torch.tensor(
                    f["fields"][t+step+1:t+step+2, :N_CHANNELS, :720, :],
                    device=DEVICE, dtype=torch.float32
                )

                # Metrics
                rmse = weighted_rmse_channels(phys_pred, phys_tgt)
                acc  = weighted_acc_channels(phys_pred, phys_tgt)

                z500_rmse_all.append(rmse[Z500_IDX].item())
                t850_rmse_all.append(rmse[T850_IDX].item())
                acc_all.append(acc.mean().item())

    print("\n--- Final Benchmark Report (Notebook-Aligned) ---")
    print(f"Weight Path: {weight_path}")
    print(f"Avg Z500 RMSE: {np.mean(z500_rmse_all):.2f} m²/s²")
    print(f"Avg T850 RMSE: {np.mean(t850_rmse_all):.2f} K")
    print(f"Avg Mean ACC:  {np.mean(acc_all):.4f}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    params = {
        "patch_size": 8,
        "embed_dim": 768,
        "depth": 12,
        "num_blocks": 8,
        "N_in_channels": 20,
        "N_out_channels": 20,
        "img_size": (720, 1440),
    }

    class ParamsObject:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            self.in_channels = 20
            self.out_channels = 20
            self.patch_size = 8
            self.num_blocks = 8

    params_obj = ParamsObject(**params)

    run_evaluation(
        params_obj,
        weight_path="weights/backbone.ckpt",
        data_path="data/out_of_sample/2018.h5",
        num_steps=4      # 24 hours
    )
