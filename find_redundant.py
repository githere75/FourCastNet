import torch
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from networks.afnonet import AFNONet

def get_rsa_with_heatmap(params_obj, weight_path, data_path, output_img="fcn_layer_similarity.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize and Load Model
    model = AFNONet(params_obj).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint)
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict)
    model.eval()

    # 2. Capture Embeddings
    layer_outputs = []
    def hook_fn(module, input, output):
        layer_outputs.append(output.detach().flatten(1))

    hooks = [block.register_forward_hook(hook_fn) for block in model.blocks]

    # 3. Load Sample Data
    means = np.load("data/stats/global_means.npy")[0, :20].reshape(20, 1, 1)
    stds = np.load("data/stats/global_stds.npy")[0, :20].reshape(20, 1, 1)
    with h5py.File(data_path, 'r') as f:
        raw_data = f['fields'][0:1, :20, :720, :1440]
        input_tensor = torch.from_numpy((raw_data - means) / stds).to(device).float()

    with torch.no_grad():
        _ = model(input_tensor)

    # 4. Compute Full Similarity Matrix
    num_layers = len(layer_outputs)
    sim_matrix = np.zeros((num_layers, num_layers))

    for i in range(num_layers):
        for j in range(num_layers):
            sim = F.cosine_similarity(layer_outputs[i], layer_outputs[j], dim=1).mean().item()
            sim_matrix[i, j] = sim

    # 5. Generate and Save Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="YlGnBu", 
                xticklabels=[f"L{i}" for i in range(num_layers)],
                yticklabels=[f"L{i}" for i in range(num_layers)])
    
    plt.title("FourCastNet Layer-wise Representation Similarity Matrix")
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.savefig(output_img)
    print(f"\nHeatmap saved successfully as {output_img}")

    # Print adjacent pairs for quick reference
    print("\nAdjacent Layer Similarities:")
    for i in range(num_layers - 1):
        print(f"Layer {i} vs Layer {i+1}: {sim_matrix[i, i+1]:.6f}")

    for h in hooks: h.remove()

if __name__ == "__main__":
    params = {
        'patch_size': 8, 'embed_dim': 768, 'depth': 12, 'num_blocks': 8,
        'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
    }

    class ParamsObject:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            self.in_channels = entries.get('N_in_channels')
            self.out_channels = entries.get('N_out_channels')
            self.patch_size = entries.get('patch_size')
            self.num_blocks = entries.get('num_blocks')

    params_obj = ParamsObject(**params)
    get_rsa_with_heatmap(params_obj, "weights/backbone.ckpt", "data/out_of_sample/2018.h5")