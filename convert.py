import torch
from safetensors.torch import save_file

# Path to your existing file
pt_path = "/root/autodl-tmp/checkpoints/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt"
# Path where the worker expects it
st_path = "/root/autodl-tmp/checkpoints/checkpoints/global_step_30000/actor/model_state_dict/model.safetensors"

# Load and convert
state_dict = torch.load(pt_path, map_location="cpu")
save_file(state_dict, st_path)

print(f"Successfully converted {pt_path} to {st_path}")