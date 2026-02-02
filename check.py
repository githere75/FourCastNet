import torch
import numpy as np
import h5py
from tqdm import tqdm
from networks.afnonet import AFNONet
from utils.YParams import YParams
from utils.weighted_acc_rmse import (
    weighted_rmse_torch_channels,
    weighted_acc_torch_channels,
)

# --------------------------------------------------
# Utility: count parameters
# --------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# --------------------------------------------------
# Utility: multi-step rollout
# --------------------------------------------------
@torch.no_grad()
def autoregressive_rollout(model, x0, steps):
    """
    x0: [1, C, H, W]
    returns list of predictions for each step
    """
    preds = []
    x = x0
    for _ in range(steps):
        x = model(x)
        preds.append(x)
    return preds

# --------------------------------------------------
# Main evaluation
# --------------------------------------------------
def evaluate_2018_multistep(
    config_path,
    config_name,
    weight_path,
    data_path,
    num_layers,
    device="cuda",
):
    # ------------------------------
    # Load config
    # ------------------------------
    params = YParams(config_path, config_name)
    params["n_layers"] = num_layers

    # ------------------------------
    # Build model
    # ------------------------------
    model = AFNONet(params).to(device)

    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    # ------------------------------
    # Metrics containers
    # ------------------------------
    lead_steps = {
        "+6h": 1,
        "+12h": 2,
        "+18h": 3,
        "+24h": 4,
    }

    rmse_scores = {k: [] for k in lead_steps}
    acc_scores = {k: [] for k in lead_steps}

    # ------------------------------
    # Load full 2018 dataset
    # ------------------------------
    with h5py.File(data_path, "r") as f:
        fields = f["fields"][:]  # [T, C, 720, 1440]

    T = fields.shape[0]

    # ------------------------------
    # Loop over entire year
    # ------------------------------
    print("Running multi-step rollout over 2018...")
    for t in tqdm(range(T - max(lead_steps.values()))):
        x0 = torch.from_numpy(fields[t:t+1]).to(device, dtype=torch.float)

        preds = autoregressive_rollout(
            model,
            x0,
            steps=max(lead_steps.values()),
        )

        for name, step in lead_steps.items():
            pred = preds[step - 1]
            gt = torch.from_numpy(fields[t + step:t + step + 1]).to(
                device, dtype=torch.float
            )

            rmse = weighted_rmse_torch_channels(pred, gt).mean().item()
            acc = weighted_acc_torch_channels(pred, gt).mean().item()

            rmse_scores[name].append(rmse)
            acc_scores[name].append(acc)

    # ------------------------------
    # Final reporting
    # ------------------------------
    print("\n" + "=" * 50)
    print(f"FOURCASTNET MULTI-STEP RESULTS (2018)")
    print(f"Model depth: {num_layers}")
    print(f"Parameters: {count_parameters(model)/1e6:.2f} M")
    print("=" * 50)

    for name in lead_steps:
        print(
            f"{name:<6} | "
            f"RMSE: {np.mean(rmse_scores[name]):.4f} | "
            f"ACC: {np.mean(acc_scores[name]):.4f}"
        )

    print("=" * 50)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    evaluate_2018_multistep(
        config_path="config/AFNO.yaml",
        config_name="afno_backbone",
        weight_path="weights/backbone.ckpt",
        data_path="data/out_of_sample/2018.h5",
        num_layers=12,   # baseline
    )
