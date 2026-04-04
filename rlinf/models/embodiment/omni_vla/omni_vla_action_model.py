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

import math
import random
from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
import torch
from omni_vla import transforms as _transforms
from omni_vla.models import model as _model
from omni_vla.models_pytorch.omni_config import OmniConfig
from omni_vla.models_pytorch.omni_vla import OmniVLA, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor


@dataclass(frozen=True)
class OmniVLAConfig(OmniConfig):
    # config for rl
    config_name: str = "omni_vla"  
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    freeze_vggt: bool = False
    freeze_spatial_expert: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token

    # ===== DSRL-specific parameters =====
    use_dsrl: bool = False  # Enable DSRL algorithm
    dsrl_state_dim: int = 8  # Raw state dimension for DSRL encoders
    dsrl_action_noise_dim: int = 32  # Noise dimension output by GaussianPolicy
    dsrl_num_q_heads: int = 10  # Number of Q-networks
    dsrl_agg_q: str = "mean"  # Q aggregation method: 'mean' | 'min'
    dsrl_image_latent_dim: int = 64  # Latent dim for lightweight image encoder
    dsrl_state_latent_dim: int = 64  # Hidden dim for state encoder
    dsrl_hidden_dims: tuple = field(
        default_factory=lambda: (128, 128, 128)
    )  # Hidden dims for Q-head and GaussianPolicy


class OmniVLAForRLActionPrediction(OmniVLA, BasePolicy):
    """
    Omni VLA model for reinforcement learning action prediction.
    """

    config: OmniVLAConfig

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    def __init__(
        self,
        config: OmniVLAConfig,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        # OmniVLA takes config and device, but parent init sets device to None
        # We need to call parent init correctly.
        # However, OmniVLA init does not use device for creating parameters except for some utility functions
        # Let's check OmniVLA init again.
        # It calls super().__init__() (nn.Module).
        # It calls self.reasoning_spatial_expert = ...
        # It calls self.spatial_to_reasoning = ...
        # It seems it doesn't strictly require device during init if we don't pass it.
        # But OmniVLA signature is __init__(self, config: OmniConfig, device: torch.device)
        # We should pass a dummy device or rely on PyTorch default (cpu).
        super().__init__(config, device=torch.device("cpu"))
        
        self.sample_actions = sample_actions_func
        self.logger = get_logger()
        self.global_step = 0
        # assert
        assert self.config.noise_method == "flow_sde", (
            f"OmniVLA only supports noise_method='flow_sde', got '{self.config.noise_method}'. "
            "flow_noise and flow_cps are not yet implemented for OmniVLA."
        )
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        if self.use_vlm_value:
            raise NotImplementedError(
                "value_after_vlm is not yet supported for OmniVLA. "
                "Set value_after_vlm=False and use action-expert based value estimation instead."
            )

        # DSRL not fully implemented for OmniVLA yet, placeholder if needed
        if self.config.use_dsrl:
            raise NotImplementedError("DSRL is not yet supported for OmniVLA")

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def _resolve_omni_vla_semantics(
        self,
        *,
        prefix_middle_no_grad_override=None,
        clone_past_key_values_override=None,
    ) -> tuple[bool, bool]:
        """Resolve behavior-semantics switches without relying only on self.training."""
        prefix_middle_no_grad = (
            self.training
            if prefix_middle_no_grad_override is None
            else bool(prefix_middle_no_grad_override)
        )
        clone_past_key_values = (
            (not self.training)
            if clone_past_key_values_override is None
            else bool(clone_past_key_values_override)
        )
        return prefix_middle_no_grad, clone_past_key_values

    @contextmanager
    def _temporary_gradient_checkpointing(self, enabled: bool | None):
        """Temporarily override gradient-checkpointing/cache flags for semantic probes."""
        if enabled is None:
            yield
            return

        reasoning_lm = self.reasoning_spatial_expert.reasoning_expert.language_model
        vision_tower = self.reasoning_spatial_expert.reasoning_expert.vision_tower
        spatial_model = self.reasoning_spatial_expert.spatial_expert.model
        action_model = self.reasoning_spatial_expert.action_expert.model

        snapshot = {
            "top_level_gc": getattr(self, "gradient_checkpointing_enabled", None),
            "reasoning_gc": getattr(reasoning_lm, "gradient_checkpointing", None),
            "vision_gc": getattr(vision_tower, "gradient_checkpointing", None),
            "spatial_gc": getattr(spatial_model, "gradient_checkpointing", None),
            "action_gc": getattr(action_model, "gradient_checkpointing", None),
            "reasoning_use_cache": getattr(reasoning_lm.config, "use_cache", None),
            "spatial_use_cache": getattr(spatial_model.config, "use_cache", None),
            "action_use_cache": getattr(action_model.config, "use_cache", None),
        }

        try:
            if snapshot["top_level_gc"] is not None:
                self.gradient_checkpointing_enabled = enabled
            if snapshot["reasoning_gc"] is not None:
                reasoning_lm.gradient_checkpointing = enabled
            if snapshot["vision_gc"] is not None:
                vision_tower.gradient_checkpointing = enabled
            if snapshot["spatial_gc"] is not None:
                spatial_model.gradient_checkpointing = enabled
            if snapshot["action_gc"] is not None:
                action_model.gradient_checkpointing = enabled
            if snapshot["reasoning_use_cache"] is not None:
                reasoning_lm.config.use_cache = not enabled
            if snapshot["spatial_use_cache"] is not None:
                spatial_model.config.use_cache = not enabled
            if snapshot["action_use_cache"] is not None:
                action_model.config.use_cache = not enabled
            yield
        finally:
            if snapshot["top_level_gc"] is not None:
                self.gradient_checkpointing_enabled = snapshot["top_level_gc"]
            if snapshot["reasoning_gc"] is not None:
                reasoning_lm.gradient_checkpointing = snapshot["reasoning_gc"]
            if snapshot["vision_gc"] is not None:
                vision_tower.gradient_checkpointing = snapshot["vision_gc"]
            if snapshot["spatial_gc"] is not None:
                spatial_model.gradient_checkpointing = snapshot["spatial_gc"]
            if snapshot["action_gc"] is not None:
                action_model.gradient_checkpointing = snapshot["action_gc"]
            if snapshot["reasoning_use_cache"] is not None:
                reasoning_lm.config.use_cache = snapshot["reasoning_use_cache"]
            if snapshot["spatial_use_cache"] is not None:
                spatial_model.config.use_cache = snapshot["spatial_use_cache"]
            if snapshot["action_use_cache"] is not None:
                action_model.config.use_cache = snapshot["action_use_cache"]

    @contextmanager
    def _temporary_behavior_eval_modules(
        self,
        enabled: bool | None,
        *,
        include_vision: bool = False,
    ):
        if not enabled:
            yield
            return

        reasoning_lm = self.reasoning_spatial_expert.reasoning_expert.language_model
        vision_tower = self.reasoning_spatial_expert.reasoning_expert.vision_tower
        spatial_model = self.reasoning_spatial_expert.spatial_expert.model
        action_model = self.reasoning_spatial_expert.action_expert.model

        modules = [reasoning_lm, spatial_model, action_model]
        if include_vision:
            modules.append(vision_tower)

        snapshot = {
            "training": {id(module): bool(module.training) for module in modules},
            "gc": {
                id(module): getattr(module, "gradient_checkpointing", None)
                for module in modules
            },
            "use_cache": {
                id(module): getattr(getattr(module, "config", None), "use_cache", None)
                for module in modules
            },
        }

        try:
            for module in modules:
                module.eval()
                if snapshot["gc"][id(module)] is not None:
                    module.gradient_checkpointing = False
                config = getattr(module, "config", None)
                if config is not None and snapshot["use_cache"][id(module)] is not None:
                    config.use_cache = True
            yield
        finally:
            for module in modules:
                module.train(snapshot["training"][id(module)])
                if snapshot["gc"][id(module)] is not None:
                    module.gradient_checkpointing = snapshot["gc"][id(module)]
                config = getattr(module, "config", None)
                if config is not None and snapshot["use_cache"][id(module)] is not None:
                    config.use_cache = snapshot["use_cache"][id(module)]

    def _get_debug_runtime_context(
        self,
        *,
        prefix_middle_no_grad: bool,
        clone_past_key_values: bool,
        prefix_cache_seq_len: int,
        middle_cache_seq_len: int,
    ) -> dict[str, Any]:
        reasoning_lm = self.reasoning_spatial_expert.reasoning_expert.language_model
        spatial_model = self.reasoning_spatial_expert.spatial_expert.model
        action_model = self.reasoning_spatial_expert.action_expert.model
        return {
            "model_training": bool(self.training),
            "gradient_checkpointing_enabled": bool(
                getattr(self, "gradient_checkpointing_enabled", False)
            ),
            "reasoning_training": bool(reasoning_lm.training),
            "spatial_training": bool(spatial_model.training),
            "action_training": bool(action_model.training),
            "prefix_middle_no_grad": bool(prefix_middle_no_grad),
            "clone_past_key_values": bool(clone_past_key_values),
            "reasoning_use_cache": bool(getattr(reasoning_lm.config, "use_cache", False)),
            "spatial_use_cache": bool(getattr(spatial_model.config, "use_cache", False)),
            "action_use_cache": bool(getattr(action_model.config, "use_cache", False)),
            "prefix_cache_seq_len": int(prefix_cache_seq_len),
            "middle_cache_seq_len": int(middle_cache_seq_len),
        }

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return super().forward(observation, actions)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        prefix_middle_no_grad_override = kwargs.get(
            "prefix_middle_no_grad_override", None
        )
        clone_past_key_values_override = kwargs.get(
            "clone_past_key_values_override", None
        )
        gradient_checkpointing_override = kwargs.get(
            "gradient_checkpointing_override", None
        )
        behavior_eval_override = kwargs.get("behavior_eval_override", None)
        behavior_eval_include_vision = bool(
            kwargs.get("behavior_eval_include_vision", False)
        )
        debug_chain_trace = bool(kwargs.get("debug_chain_trace", False))
        debug_train_rollout_semantics_gap = bool(
            kwargs.get("debug_train_rollout_semantics_gap", False)
        )
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        observation_forward_inputs = {
            key: value
            for key, value in forward_inputs.items()
            if not key.startswith("debug_")
        }
        # input transform
        observation = self.input_transform(observation_forward_inputs, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        with self._temporary_behavior_eval_modules(
            behavior_eval_override,
            include_vision=behavior_eval_include_vision,
        ):
            with self._temporary_gradient_checkpointing(
                gradient_checkpointing_override
            ):
                log_prob_outputs = self.get_log_prob_value(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    chains,
                    denoise_inds,
                    compute_values,
                    prefix_middle_no_grad_override=prefix_middle_no_grad_override,
                    clone_past_key_values_override=clone_past_key_values_override,
                    return_debug_trace=debug_chain_trace,
                )
        if debug_chain_trace:
            log_probs, value_t, entropy, debug_trace_context = log_prob_outputs
        else:
            log_probs, value_t, entropy = log_prob_outputs
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=True)
        output_dict = {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }
        if debug_chain_trace:
            output_dict["debug_trace_context"] = debug_trace_context
        if debug_train_rollout_semantics_gap:
            # If the current forward is already using behavior/eval semantics, avoid a
            # second expensive rollout-style recompute. This path was causing OOM once
            # current train forward had been aligned to rollout semantics.
            if behavior_eval_override:
                semantic_gap = torch.zeros_like(log_probs.detach().float())
            else:
                with torch.no_grad(), self._temporary_behavior_eval_modules(
                    True,
                    include_vision=behavior_eval_include_vision,
                ), self._temporary_gradient_checkpointing(False):
                    rollout_semantic_log_probs, _, _ = self.get_log_prob_value(
                        images,
                        img_masks,
                        lang_tokens,
                        lang_masks,
                        state,
                        chains,
                        denoise_inds,
                        False,
                        prefix_middle_no_grad_override=False,
                        clone_past_key_values_override=True,
                    )
                    rollout_semantic_log_probs = rollout_semantic_log_probs[
                        :, :, : self.config.action_chunk, : self.config.action_env_dim
                    ].mean(dim=1)
                    semantic_gap = (
                        log_probs.detach().float() - rollout_semantic_log_probs.float()
                    )
            output_dict["debug_metrics"] = {
                "actor/debug_train_rollout_logprob_gap_mean": semantic_gap.mean(),
                "actor/debug_train_rollout_logprob_gap_abs_mean": semantic_gap.abs().mean(),
                "actor/debug_train_rollout_logprob_gap_max": semantic_gap.abs().max(),
            }
        return output_dict

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation
        processed_obs["observation/state"] = env_obs["states"]
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)

        # Non-DSRL or eval mode
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        raw_actions = outputs["actions"]
        # if self.global_step < 3:
        #     self.logger.info(
        #         f"[OmniVLA DEBUG] raw_actions stats: "
        #         f"mean={raw_actions.mean().item():.4f}, std={raw_actions.std().item():.4f}, "
        #         f"min={raw_actions.min().item():.4f}, max={raw_actions.max().item():.4f}, "
        #         f"shape={tuple(raw_actions.shape)}"
        #     )
        actions = self.output_transform(
            {"actions": raw_actions, "state": observation.state}
        )["actions"].numpy()
        # if self.global_step < 3:
        #     self.logger.info(
        #         f"[OmniVLA DEBUG] final_actions stats: "
        #         f"mean={actions.mean():.4f}, std={actions.std():.4f}, "
        #         f"min={actions.min():.4f}, max={actions.max():.4f}, "
        #         f"shape={actions.shape}"
        #     )
        prev_logprobs = outputs["prev_logprobs"]
        prev_values = outputs["prev_values"]
        forward_action = None

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        if forward_action is not None:
            forward_inputs["action"] = forward_action

        # Clone observations to avoid cross-step reference issues.
        cloned_obs = copy_dict_tensor(
            {k: v for k, v in to_process_obs.items() if k != "prompt"}
        )
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(self.action_in_proj.weight.dtype)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self.get_position_ids(prefix_pad_masks)

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        # Access config for eager attention
        self.reasoning_spatial_expert.reasoning_expert.language_model.config._attn_implementation = "eager" 
        self.reasoning_spatial_expert.spatial_expert.config._attn_implementation = "eager"
        self.reasoning_spatial_expert.action_expert.config._attn_implementation = "eager"

        # Undo the normalizer scaling inside GemmaModel during inference to align with training logic
        normalizer = torch.tensor(prefix_embs.shape[-1]**0.5, dtype=prefix_embs.dtype, device=prefix_embs.device)
        prefix_embs_unscaled = prefix_embs / normalizer

        # reasoning_spatial_expert.forward returns ([prefix_out, middle_out, suffix_out], past_key_values)
        prefix_middle_no_grad = self.training and self.config.train_expert_only

        if prefix_middle_no_grad:
            with torch.no_grad():
                _, past_key_values = self.reasoning_spatial_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs_unscaled, None, None],
                    use_cache=True,
                )
        else:
            _, past_key_values = self.reasoning_spatial_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_unscaled, None, None],
                use_cache=True,
            )

        # 2. Process middle (spatial features)
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(
            images, img_masks
        )

        middle_len = middle_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]

        # Build attention mask for prefix + middle
        prefix_pad_2d_masks = middle_pad_masks[:, :, None] & prefix_pad_masks[:, None, :]
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values
        middle_position_ids_2d = torch.arange(1, middle_len + 1, dtype=torch.long, device=device)
        middle_position_ids_2d = middle_position_ids_2d.unsqueeze(0).expand(batch_size, -1) 
        middle_position_ids_2d = middle_position_ids_2d + max_prefix_position_ids 

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        normalizer_m = torch.tensor(middle_embs.shape[-1]**0.5, dtype=middle_embs.dtype, device=middle_embs.device)
        middle_embs_unscaled = middle_embs / normalizer_m

        # Process middle, reuse prefix's KV cache
        if prefix_middle_no_grad:
            with torch.no_grad():
                (_, _, _), past_key_values = self.reasoning_spatial_expert.forward(
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=middle_position_ids_2d,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, middle_embs_unscaled, None],
                    use_cache=True,
                )
            if past_key_values is not None:
                past_key_values.key_cache = [k.detach() for k in past_key_values.key_cache]
                past_key_values.value_cache = [v.detach() for v in past_key_values.value_cache]
        else:
            (_, _, _), past_key_values = self.reasoning_spatial_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=middle_position_ids_2d,
                past_key_values=past_key_values,
                inputs_embeds=[None, middle_embs_unscaled, None],
                use_cache=True,
            )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks, # Note: this should be curr_pad_masks (prefix+middle)
                past_key_values,
                sample_mode,
                self.config.num_steps,
                compute_values,
                middle_pad_masks=middle_pad_masks, # Pass middle pad masks
                max_position_ids=middle_position_ids_2d.max(dim=-1, keepdim=True).values # Pass max pos
            )
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        
        values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks, # This argument name is kept for compatibility but we need to handle it correctly
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
        middle_pad_masks=None,
        max_position_ids=None,
        clone_past_key_values_override=None,
    ):
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        
        # ... noise annealing logic same as OpenPi0 ...
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.config.noise_level).to(device)
        
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        # In OmniVLA, we need prefix_pad_masks AND middle_pad_masks to form curr_pad_masks
        if middle_pad_masks is not None:
             curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)
        else:
             curr_pad_masks = prefix_pad_masks # Fallback

        # Use OmniVLA's denoise_step logic but adapted to return suffix_out first
        # OmniVLA.denoise_step returns v_t directly. We need suffix_out for value head.
        
        # We need to reimplement denoise_step logic here to get suffix_out
        expanded_time = t_input.expand(bsize).to(self.action_in_proj.weight.dtype)
        
        # 1) Only generate suffix embedding
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, expanded_time
        )
        if (
            self.reasoning_spatial_expert.reasoning_expert.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        suffix_len = suffix_pad_masks.shape[1]
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        
        if past_key_values is not None:
            cached_seq_len = past_key_values.get_seq_length()
            _, clone_past_key_values = self._resolve_omni_vla_semantics(
                clone_past_key_values_override=clone_past_key_values_override
            )
            if not clone_past_key_values:
                past_key_values_for_suffix = past_key_values
            else:
                from transformers.cache_utils import DynamicCache
                past_key_values_for_suffix = DynamicCache()
                past_key_values_for_suffix.key_cache = list(past_key_values.key_cache)
                past_key_values_for_suffix.value_cache = list(past_key_values.value_cache)
                if hasattr(past_key_values, "_seen_tokens"):
                    past_key_values_for_suffix._seen_tokens = past_key_values._seen_tokens
        else:
            cached_seq_len = 0
            past_key_values_for_suffix = None

        if cached_seq_len > 0:
            # We need full prefix mask (prefix + middle). 
            # cached_seq_len corresponds to prefix + middle length.
            # curr_pad_masks passed in should match cached_seq_len ideally.
            cached_mask = suffix_pad_masks[:, :, None] & curr_pad_masks[:, None, :cached_seq_len]
            full_att_2d_masks = torch.cat([cached_mask, suffix_att_2d_masks], dim=2)
        else:
            full_att_2d_masks = suffix_att_2d_masks

        if max_position_ids is None:
             # Should not happen in correct usage
             max_position_ids = torch.zeros((bsize, 1), dtype=torch.long, device=device)

        position_ids = torch.arange(1, suffix_len + 1, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(bsize, -1)
        position_ids = position_ids + max_position_ids

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        normalizer_s = torch.tensor(suffix_embs.shape[-1]**0.5, dtype=suffix_embs.dtype, device=suffix_embs.device)
        suffix_embs_unscaled = suffix_embs / normalizer_s

        outputs_embeds, _ = self.reasoning_spatial_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values_for_suffix,
            inputs_embeds=[None, None, suffix_embs_unscaled],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)

        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)

        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        
        # ... weights calculation same as OpenPi0 ...
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
             # Assuming flow_sde for now as default
             # ... copy logic ...
             if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
             else:
                raise ValueError(
                    f"Unsupported noise_method '{self.config.noise_method}' in OmniVLA. "
                    "Only 'flow_sde' is supported."
                )

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
        prefix_middle_no_grad_override=None,
        clone_past_key_values_override=None,
        return_debug_trace=False,
    ):
        bsize = state.shape[0]
        prefix_middle_no_grad, clone_past_key_values = self._resolve_omni_vla_semantics(
            prefix_middle_no_grad_override=prefix_middle_no_grad_override,
            clone_past_key_values_override=clone_past_key_values_override,
        )
        prefix_cache_seq_len = 0
        middle_cache_seq_len = 0

        # 1. Prefix
        if prefix_middle_no_grad:
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self.get_position_ids(prefix_pad_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.reasoning_spatial_expert.reasoning_expert.language_model.config._attn_implementation = "eager"
        self.reasoning_spatial_expert.spatial_expert.config._attn_implementation = "eager"
        self.reasoning_spatial_expert.action_expert.config._attn_implementation = "eager"

        normalizer = torch.tensor(prefix_embs.shape[-1]**0.5, dtype=prefix_embs.dtype, device=prefix_embs.device)
        prefix_embs_unscaled = prefix_embs / normalizer

        if prefix_middle_no_grad:
            with torch.no_grad():
                _, past_key_values = self.reasoning_spatial_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs_unscaled, None, None],
                    use_cache=True,
                )
        else:
            _, past_key_values = self.reasoning_spatial_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_unscaled, None, None],
                use_cache=True,
            )
        if past_key_values is not None:
            prefix_cache_seq_len = int(past_key_values.get_seq_length())

        # 2. Middle
        if prefix_middle_no_grad:
            with torch.no_grad():
                middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(
                    images, img_masks
                )
        else:
            middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(
                images, img_masks
            )
        middle_len = middle_pad_masks.shape[1]
        device = state.device
        
        prefix_pad_2d_masks = middle_pad_masks[:, :, None] & prefix_pad_masks[:, None, :]
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values
        middle_position_ids_2d = torch.arange(1, middle_len + 1, dtype=torch.long, device=device)
        middle_position_ids_2d = middle_position_ids_2d.unsqueeze(0).expand(bsize, -1) 
        middle_position_ids_2d = middle_position_ids_2d + max_prefix_position_ids 

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        normalizer_m = torch.tensor(middle_embs.shape[-1]**0.5, dtype=middle_embs.dtype, device=middle_embs.device)
        middle_embs_unscaled = middle_embs / normalizer_m

        if prefix_middle_no_grad:
            with torch.no_grad():
                (_, _, _), past_key_values = self.reasoning_spatial_expert.forward(
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=middle_position_ids_2d,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, middle_embs_unscaled, None],
                    use_cache=True,
                )
            if past_key_values is not None:
                past_key_values.key_cache = [k.detach() for k in past_key_values.key_cache]
                past_key_values.value_cache = [v.detach() for v in past_key_values.value_cache]
        else:
            (_, _, _), past_key_values = self.reasoning_spatial_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=middle_position_ids_2d,
                past_key_values=past_key_values,
                inputs_embeds=[None, middle_embs_unscaled, None],
                use_cache=True,
            )
        if past_key_values is not None:
            middle_cache_seq_len = int(past_key_values.get_seq_length())

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        
        max_position_ids = middle_position_ids_2d.max(dim=-1, keepdim=True).values

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
                middle_pad_masks=middle_pad_masks,
                max_position_ids=max_position_ids,
                clone_past_key_values_override=clone_past_key_values_override,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if not self.use_vlm_value:
                chains_values.append(value_t)
        
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        chains_entropy = torch.stack(chains_entropy, dim=1)

        if return_debug_trace:
            return (
                chains_log_probs,
                chains_values,
                chains_entropy,
                self._get_debug_runtime_context(
                    prefix_middle_no_grad=prefix_middle_no_grad,
                    clone_past_key_values=clone_past_key_values,
                    prefix_cache_seq_len=prefix_cache_seq_len,
                    middle_cache_seq_len=middle_cache_seq_len,
                ),
            )
        return chains_log_probs, chains_values, chains_entropy

    def get_logprob_norm(self, sample, mu, sigma):
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy