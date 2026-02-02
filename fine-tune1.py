import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from networks.afnonet import AFNONet
import h5py
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
from evaluate import run_full_year_benchmark


def latitude_weighted_mse(pred, target):
    """
    Calculates MSE weighted by the cosine of the latitude.
    pred/target shape: [Batch, Channels, Lat, Lon]
    """
    # Create latitude weights: cos(lat)
    # 720 points from 90N to 90S
    lats = torch.linspace(90, -90, 720).to(pred.device)
    weights = torch.cos(torch.deg2rad(lats)).view(1, 1, 720, 1)
    
    # Standard MSE multiplied by weights
    loss = (pred - target) ** 2
    weighted_loss = (loss * weights).mean()
    return weighted_loss


# ... [Include your Dataset class from our previous discussion] ...
class ERA5MultiYearDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.means = np.load("data/stats/global_means.npy")[0, :20].reshape(1, 20, 1, 1)
        self.stds = np.load("data/stats/global_stds.npy")[0, :20].reshape(1, 20, 1, 1)

        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        # Calculate indices for each file
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                length = f['fields'].shape[0] - 1 # -1 because we need t and t+1
                self.file_lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Determine which file the index belongs to
        file_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if idx < self.cumulative_lengths[i+1]:
                file_idx = i
                break
        
        # Local index within the specific file
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        with h5py.File(self.file_paths[file_idx], 'r', swmr=True) as f:
            x = f['fields'][local_idx, :20, :720, :]
            y = f['fields'][local_idx + 1, :20, :720, :]

        x = (x - self.means[0, :, :720, :]) / self.stds[0, :, :720, :]
        y = (y - self.means[0, :, :720, :]) / self.stds[0, :, :720, :]

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

def fine_tune_model(params_obj, pruned_ckpt_path, train_file_list, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model with Pruned Depth
    model = AFNONet(params_obj).to(device)
    
    # 2. Load the Pruned Weights
    checkpoint = torch.load(pruned_ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    # model.load_state_dict(state_dict, strict=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Load into the model
    model.load_state_dict(new_state_dict, strict=False)
    
    # 3. Training Setup
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Small LR for recovery
    scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1)

    dataset = ERA5MultiYearDataset(train_file_list) 
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    print("Starting Fine-Tuning...")

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device).float(), y.to(device).float()
            
            optimizer.zero_grad()
            prediction = model(x)
            
            # Loss calculation
            loss = latitude_weighted_mse(prediction, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/1], Step [{i}/{len(loader)}], Loss: {loss.item():.6f}")
        scheduler.step()

        # Save checkpoint after each epoch
        print(f'Running epoch {epoch+1}...')
        torch.save({'model_state': model.state_dict()}, f"weights/fine-tuned models/pruned_1_layers_epoch_{epochs}.ckpt")
        run_full_year_benchmark(params_obj, f"weights/fine-tuned models/pruned_1_layers_epoch_{epochs}.ckpt", "data/out_of_sample/2018.h5", num_steps=1)

if __name__ == "__main__":
    # Ensure depth matches your pruned model (e.g., 9 or 11)
    params = {
        'patch_size': 8, 'embed_dim': 768, 'depth': 12, 'num_blocks': 8, 'depth': 6,
        'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
    }
    class ParamsObject:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            # Match internal naming used in afnonet.py
            self.in_channels = entries.get('N_in_channels')
            self.out_channels = entries.get('N_out_channels')
            self.patch_size = entries.get('patch_size')
            self.num_blocks = entries.get('num_blocks')

    params_obj = ParamsObject(**params)
    
    # Create list of paths for 2014 and 2015
    training_files = [
        # "data/train/2014.h5",
        "data/train/2015.h5"
    ]
    for i in range(1,7):
        fine_tune_model(params_obj, f"weights/fine-tuned models/pruned_1_layers_epoch_{epoch}.ckpt", training_files)