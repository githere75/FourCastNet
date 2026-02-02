import torch
from collections import OrderedDict

def prune_fcn_layers(input_ckpt, output_ckpt, layers_to_remove):
    """
    Removes specific AFNO blocks from the checkpoint and re-indexes the rest.
    """
    # Load original weights (handling PyTorch 2.6+ security)
    checkpoint = torch.load(input_ckpt, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint)
    
    new_state_dict = OrderedDict()
    current_new_idx = 0
    total_original_layers = 12 
    
    print(f"Pruning layers: {layers_to_remove}")

    for i in range(total_original_layers):
        if i in layers_to_remove:
            continue
        
        # Identify weights for the current block (e.g., 'blocks.5.norm1.weight')
        # We also handle the 'module.' prefix if present in your checkpoint
        old_prefix = f"blocks.{i}."
        new_prefix = f"blocks.{current_new_idx}."
        
        for key, value in state_dict.items():
            if old_prefix in key:
                # Replace the old index with the new sequential index
                new_key = key.replace(old_prefix, new_prefix)
                new_state_dict[new_key] = value
            elif f"module.{old_prefix}" in key:
                new_key = key.replace(f"module.{old_prefix}", new_prefix)
                new_state_dict[new_key] = value
        
        current_new_idx += 1
    
    # Copy non-block weights (embeddings, normalization, and head)
    for key, value in state_dict.items():
        if "blocks." not in key:
            # Strip 'module.' if it exists to make it a standard local model
            clean_key = key.replace("module.", "")
            new_state_dict[clean_key] = value

    # Save the pruned checkpoint
    torch.save(new_state_dict, output_ckpt)
    print(f"Successfully saved {current_new_idx}-layer model to {output_ckpt}")

if __name__ == "__main__":
    # Sequentially prune based on your RSA results
    prune_fcn_layers("weights/backbone.ckpt", "weights/pruned models/pruned_5_layers.ckpt", [3,4,2,1,10])