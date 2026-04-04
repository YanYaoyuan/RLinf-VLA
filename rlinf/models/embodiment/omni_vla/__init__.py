# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    import safetensors.torch

    from omni_vla.models_pytorch.omni_config import OmniConfig
    from rlinf.models.embodiment.omni_vla.omni_vla_action_model import (
        OmniVLAForRLActionPrediction,
        OmniVLAForRLConfig,
    )

    # ------------------------------------------------------------------
    # 1. Build OmniConfig from YAML
    # ------------------------------------------------------------------
    omni_cfg = getattr(cfg, "omni_vla", None) or {}

    # Base OmniConfig parameters
    omni_config_kwargs = {}

    # Map from YAML to OmniConfig fields
    config_field_map = {
        "g2vlm_path": "g2vlm_path",
        "g2vlm_config_path": "g2vlm_config_path",
        "vlm_pretrained_path": "vlm_pretrained_path",
        "vggt_pretrained_path": "vggt_pretrained_path",
        "omni_pretrained_path": "omni_pretrained_path",
        "action_dim": "action_dim",
        "action_horizon": "action_horizon",
        "pi05": "pi05",
        "freeze_vision_encoder": "freeze_vision_encoder",
        "freeze_language_model": "freeze_language_model",
        "freeze_VGGT_model": "freeze_VGGT_model",
        "train_expert_only": "train_expert_only",
        "train_vlm_only": "train_vlm_only",
        "dtype": "dtype",
        "action_expert_variant": "action_expert_variant",
        "spatial_expert_variant": "spatial_expert_variant",
        "paligemma_variant": "paligemma_variant",
        "max_state_dim": "max_state_dim",
        "max_action_dim": "max_action_dim",
    }

    for yaml_key, config_key in config_field_map.items():
        if yaml_key in omni_cfg:
            omni_config_kwargs[config_key] = omni_cfg[yaml_key]

    omni_config = OmniConfig(**omni_config_kwargs)

    # ------------------------------------------------------------------
    # 2. Build RL config
    # ------------------------------------------------------------------
    rl_config_kwargs = {}
    rl_field_map = {
        "noise_method": "noise_method",
        "noise_level": "noise_level",
        "noise_anneal": "noise_anneal",
        "noise_params": "noise_params",
        "action_chunk": "action_chunk",
        "action_env_dim": "action_env_dim",
        "num_steps": "num_steps",
        "train_expert_only": "train_expert_only",
        "safe_get_logprob": "safe_get_logprob",
        "joint_logprob": "joint_logprob",
        "ignore_last": "ignore_last",
        "detach_critic_input": "detach_critic_input",
        "chunk_critic_input": "chunk_critic_input",
        "add_value_head": "add_value_head",
        "value_after_vlm": "value_after_vlm",
        "value_vlm_mode": "value_vlm_mode",
        "num_images_in_input": "num_images_in_input",
    }

    for yaml_key, rl_key in rl_field_map.items():
        if yaml_key in omni_cfg:
            rl_config_kwargs[rl_key] = omni_cfg[yaml_key]

    # Also pick up top-level cfg overrides
    if hasattr(cfg, "add_value_head") and cfg.add_value_head:
        rl_config_kwargs["add_value_head"] = cfg.add_value_head
    if hasattr(cfg, "action_dim"):
        rl_config_kwargs["action_env_dim"] = cfg.action_dim
    if hasattr(cfg, "num_action_chunks"):
        rl_config_kwargs["action_chunk"] = cfg.num_action_chunks

    rl_config = OmniVLAForRLConfig(**rl_config_kwargs)

    # ------------------------------------------------------------------
    # 3. Create model
    # ------------------------------------------------------------------
    model = OmniVLAForRLActionPrediction(omni_config, rl_config)

    # Apply freeze strategy
    if rl_config.train_expert_only:
        model.freeze_vlm()
    model.set_requires_grad()

    # ------------------------------------------------------------------
    # 4. Load weights
    # ------------------------------------------------------------------
    checkpoint_dir = str(cfg.model_path)

    # Check for FSDP checkpoint formats
    full_weights_path = os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt")
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    if os.path.exists(full_weights_path):
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Try safetensors format
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if weight_paths:
            all_state_dict = {}
            for weight_path in weight_paths:
                state_dict = safetensors.torch.load_file(weight_path, device="cpu")
                all_state_dict.update(state_dict)
            model.load_state_dict(all_state_dict, strict=False)
        else:
            # If omni_pretrained_path is specified, load from there
            pretrained_path = omni_config.omni_pretrained_path
            if pretrained_path and os.path.exists(pretrained_path):
                pretrained_weights = sorted(
                    glob.glob(os.path.join(pretrained_path, "*.safetensors"))
                )
                if pretrained_weights:
                    all_state_dict = {}
                    for wp in pretrained_weights:
                        state_dict = safetensors.torch.load_file(wp, device="cpu")
                        all_state_dict.update(state_dict)
                    model.load_state_dict(all_state_dict, strict=False)

    # Convert to bfloat16 for selected params if needed
    if torch_dtype == torch.bfloat16:
        for name, param in model.named_parameters():
            if "norm" not in name.lower() and "ln" not in name.lower():
                param.data = param.data.to(torch.bfloat16)

    return model
