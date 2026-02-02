import torch
import yaml
import torch.nn as nn
from easydict import EasyDict
from inference.inference import setup

CONFIG_FILE = "config/AFNO.yaml"
CONFIG_NAME = "afno_backbone"
BASE_CKPT = "weights/backbone.ckpt"
PRUNED_CKPT = "weights/pruned_afno.ckpt"

KEEP_BLOCKS = [0, 1, 2, 3, 4, 5]  # example

# Load config
with open(CONFIG_FILE, "r") as f:
    cfg = yaml.unsafe_load(f)

params = EasyDict(cfg[CONFIG_NAME])
params.best_checkpoint_path = BASE_CKPT

# IMPORTANT: build model via setup()
_, model = setup(params)

print("Original blocks:", len(model.blocks))

# Prune blocks IN PLACE
model.blocks = nn.ModuleList([model.blocks[i] for i in KEEP_BLOCKS])
model.num_blocks = len(model.blocks)

print("Pruned blocks:", len(model.blocks))

# Save EXACTLY what inference.py expects
torch.save(
    {"model_state": model.state_dict()},
    PRUNED_CKPT
)

print("Saved pruned checkpoint:", PRUNED_CKPT)

# Sanity check keys
state = model.state_dict()
assert any(k.startswith("patch_embed.") for k in state.keys())
assert any(k.startswith("blocks.") for k in state.keys())
assert any(k.startswith("head.") for k in state.keys())

print("Checkpoint structure is valid")