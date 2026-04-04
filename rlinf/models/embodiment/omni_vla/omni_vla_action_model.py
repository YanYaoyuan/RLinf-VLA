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
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from omni_vla.models import model as _model
from omni_vla.models_pytorch.omni_config import OmniConfig
from omni_vla.models_pytorch.omni_vla import OmniVLA, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor


@dataclass
class OmniVLAForRLConfig:
    """RL-specific configuration for OmniVLA, layered on top of OmniConfig."""

    # Noise / sampling
    noise_method: str = "flow_sde"  # flow_ode, flow_sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps

    # Action chunking
    action_chunk: int = 10
    action_env_dim: int = 7  # actual env action dim (before padding to 32)
    num_steps: int = 10  # denoise steps

    # Training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False
    ignore_last: bool = False

    # Critic
    detach_critic_input: bool = False
    chunk_critic_input: bool = False
    add_value_head: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"

    # Image input
    num_images_in_input: int = 3  # OmniVLA uses 3 cameras


class OmniVLAForRLActionPrediction(OmniVLA, BasePolicy):
    """
    OmniVLA model wrapped for reinforcement learning action prediction.
    Follows the same interface pattern as OpenPi0ForRLActionPrediction.
    """

    def __init__(self, config: OmniConfig, rl_config: OmniVLAForRLConfig, device=None):
        if device is None:
            device = torch.device("cpu")
        super().__init__(config, device)
        self.rl_config = rl_config
        self.logger = get_logger()
        self.global_step = 0

        # Value head
        action_expert_width = self.action_out_proj.in_features
        if self.rl_config.add_value_head:
            self.value_head = ValueHead(
                input_dim=action_expert_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        self.use_vlm_value = (
            self.rl_config.value_after_vlm and self.rl_config.add_value_head
        )

        # FSDP metadata
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(
                module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name
            )

    @property
    def _no_split_modules(self) -> list[str]:
        if self.rl_config.train_expert_only:
            return [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
                "Qwen2DecoderLayer",
            ]
        return [
            "GemmaMLP",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaRotaryEmbedding",
            "Qwen2DecoderLayer",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "spatial_to_reasoning",
        ]

    def set_global_step(self, global_step):
        self.global_step = global_step

    def freeze_vlm(self):
        """Freeze the VLM (reasoning expert) parameters."""
        for param in self.reasoning_spatial_expert.reasoning_expert.parameters():
            param.requires_grad = False
        self.logger.info("Froze reasoning expert (VLM) parameters")

    # ------------------------------------------------------------------
    # Observation processing (env_obs -> model inputs)
    # ------------------------------------------------------------------

    def obs_processor(self, env_obs):
        """Convert RLinf environment observations to OmniVLA Observation format.

        env_obs keys:
            - main_images: [B, C, H, W] primary camera
            - wrist_images: [B, C, H, W] or None
            - extra_view_images: [B, C, H, W] or None
            - states: [B, state_dim]
            - task_descriptions: list[str]
        """
        images = {}
        image_masks = {}
        main = env_obs["main_images"]
        bsize = main.shape[0]
        device = main.device

        # Primary camera -> base_0_rgb
        images["base_0_rgb"] = main
        image_masks["base_0_rgb"] = torch.ones(bsize, dtype=torch.bool, device=device)

        # Wrist camera -> left_wrist_0_rgb
        if env_obs.get("wrist_images") is not None:
            images["left_wrist_0_rgb"] = env_obs["wrist_images"]
            image_masks["left_wrist_0_rgb"] = torch.ones(bsize, dtype=torch.bool, device=device)
        else:
            images["left_wrist_0_rgb"] = torch.zeros_like(main)
            image_masks["left_wrist_0_rgb"] = torch.zeros(bsize, dtype=torch.bool, device=device)

        # Extra view -> right_wrist_0_rgb
        if env_obs.get("extra_view_images") is not None:
            images["right_wrist_0_rgb"] = env_obs["extra_view_images"]
            image_masks["right_wrist_0_rgb"] = torch.ones(bsize, dtype=torch.bool, device=device)
        else:
            images["right_wrist_0_rgb"] = torch.zeros_like(main)
            image_masks["right_wrist_0_rgb"] = torch.zeros(bsize, dtype=torch.bool, device=device)

        # State - pad to max_state_dim (32)
        state = env_obs["states"]
        if state.shape[-1] < self.config.max_state_dim:
            state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))

        processed = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }
        return processed, env_obs.get("task_descriptions")

    def prepare_observation(self, processed_obs):
        """Convert processed obs dict to OmniVLA Observation object."""
        return _model.Observation.from_dict(processed_obs)

    def precision_processor(self, observation):
        """Move observation to model device."""
        device = next(self.parameters()).device
        import jax

        return jax.tree.map(
            lambda x: x.to(device=device).contiguous() if torch.is_tensor(x) else x,
            observation,
        )

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.NFT:
            return self.forward_nft(**kwargs)
        else:
            raise NotImplementedError(f"Forward type {forward_type} not supported")

    # ------------------------------------------------------------------
    # SFT forward (supervised fine-tuning)
    # ------------------------------------------------------------------

    def sft_forward(self, data, **kwargs):
        if hasattr(self, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable()
        observation = data["observation"]
        actions = data["actions"]
        return OmniVLA.forward(self, observation, actions)

    # ------------------------------------------------------------------
    # Default forward (RL: compute log_prob, value, entropy)
    # ------------------------------------------------------------------

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]

        obs_dict = self._extract_obs_from_forward_inputs(forward_inputs)
        observation = self.prepare_observation(obs_dict)
        observation = self.precision_processor(observation)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [m.to(device) for m in img_masks]
        state = state.to(device)

        log_probs, value_t, entropy = self.get_log_prob_value(
            images, img_masks, lang_tokens, lang_masks, state,
            chains, denoise_inds, compute_values,
        )

        log_probs = log_probs[
            :, :, : self.rl_config.action_chunk, : self.rl_config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.rl_config.action_chunk, : self.rl_config.action_env_dim
        ]

        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------
    # NFT forward
    # ------------------------------------------------------------------

    def forward_nft(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        obs_dict = self._extract_obs_from_forward_inputs(forward_inputs)
        observation = self.prepare_observation(obs_dict)
        observation = self.precision_processor(observation)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [m.to(device) for m in img_masks]
        state = state.to(device)

        nft_explicit_inputs = kwargs.get("nft_explicit_inputs", None)
        if nft_explicit_inputs is not None:
            x_t = nft_explicit_inputs["x_t"]
            t = nft_explicit_inputs["timesteps"]
        else:
            if "chains" not in forward_inputs:
                raise ValueError("forward_nft requires `chains` or `nft_explicit_inputs`.")
            x_0 = forward_inputs["chains"][:, -1].to(device)
            bsize = x_0.shape[0]
            t = torch.rand((bsize,), device=device)
            t_expanded = t[:, None, None]
            noise = torch.randn_like(x_0)
            x_t = (1 - t_expanded) * x_0 + t_expanded * noise

        v_t, suffix_out = self._get_velocity_full_forward(
            images, img_masks, lang_tokens, lang_masks, state, x_t, t
        )
        v_t = v_t[:, : self.rl_config.action_chunk, :]

        bsize = x_t.shape[0]
        compute_values = kwargs.get("compute_values", False)
        result: dict[str, Any] = {"v_theta": v_t, "x_t": x_t, "timesteps": t}
        if compute_values and self.rl_config.add_value_head:
            result["values"] = self._compute_value_from_suffix(suffix_out)[:, None]
        return result

    # ------------------------------------------------------------------
    # predict_action_batch (rollout inference)
    # ------------------------------------------------------------------

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        processed_obs, prompts = self.obs_processor(env_obs)

        if prompts is not None and "tokenized_prompt" not in processed_obs:
            processed_obs = self._tokenize_prompts(processed_obs, prompts)

        observation = self.prepare_observation(processed_obs)
        observation = self.precision_processor(observation)

        outputs = self._sample_actions_rl(
            observation, mode=mode, compute_values=compute_values
        )

        raw_actions = outputs["actions"]
        actions = raw_actions[
            :, : self.rl_config.action_chunk, : self.rl_config.action_env_dim
        ]
        actions = actions.reshape(actions.shape[0], -1)

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
        }
        if "tokenized_prompt" in processed_obs:
            forward_inputs["tokenized_prompt"] = processed_obs["tokenized_prompt"]
        if "tokenized_prompt_mask" in processed_obs:
            forward_inputs["tokenized_prompt_mask"] = processed_obs["tokenized_prompt_mask"]

        forward_inputs["action"] = actions.contiguous()
        forward_inputs["model_action"] = (
            raw_actions.reshape(raw_actions.shape[0], -1).contiguous()
        )

        # Clone observations for replay buffer
        cloned_obs = {}
        for k, v in processed_obs.items():
            if k in ("image", "image_mask"):
                cloned_obs[k] = {
                    sk: sv.detach().clone() if torch.is_tensor(sv) else sv
                    for sk, sv in v.items()
                }
            elif torch.is_tensor(v):
                cloned_obs[k] = v.detach().clone()
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        return actions, result

    # ------------------------------------------------------------------
    # DAgger SFT batch preparation
    # ------------------------------------------------------------------

    def prepare_dagger_sft_batch(self, batch):
        device = next(self.parameters()).device
        bsz = batch["action"].shape[0]

        obs_dict = self._extract_obs_from_forward_inputs(batch)
        observation = self.prepare_observation(obs_dict)
        observation = self.precision_processor(observation)

        if "model_action" in batch:
            actions = (
                batch["model_action"]
                .reshape(bsz, self.config.action_horizon, self.config.action_dim)
                .clone()
            )
        else:
            actions = batch["action"].reshape(bsz, self.rl_config.action_chunk, -1)
            if actions.shape[-1] < self.config.action_dim:
                actions = F.pad(actions, (0, self.config.action_dim - actions.shape[-1]))
            if actions.shape[1] < self.config.action_horizon:
                actions = F.pad(actions, (0, 0, 0, self.config.action_horizon - actions.shape[1]))

        import jax

        observation = jax.tree.map(
            lambda x: torch.as_tensor(x, device=device).contiguous().clone()
            if torch.is_tensor(x) else x,
            observation,
        )
        return {
            "observation": observation,
            "actions": actions.to(torch.float32).to(device),
        }

    # ------------------------------------------------------------------
    # Internal: RL-aware sample_actions (with chains, log_probs)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_actions_rl(
        self,
        observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> dict[str, Any]:
        """Sample actions with chain recording for RL training."""
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        bsize = state.shape[0]
        device = state.device
        num_steps = self.rl_config.num_steps

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Build prefix + middle KV cache
        combined_pad_masks, past_key_values, max_position_ids = (
            self._build_prefix_middle_cache(images, img_masks, lang_tokens, lang_masks)
        )

        target_dtype = self.action_in_proj.weight.dtype
        x_t = noise.to(target_dtype)

        chains = [x_t]
        log_probs = []
        values = []

        # Denoise index selection
        if mode == "train":
            if self.rl_config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.rl_config.ignore_last:
                    idx = random.randint(0, num_steps - 2)
                else:
                    idx = random.randint(0, num_steps - 1)
                denoise_inds = torch.tensor([idx] * num_steps)
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        if self.rl_config.joint_logprob:
            initial_log_prob = self._get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # Denoise loop
        for step_idx in range(num_steps):
            if step_idx == denoise_inds[0][step_idx]:
                sample_method = self.rl_config.noise_method
            else:
                sample_method = "flow_ode"

            x_t_mean, x_t_std, value_t, v_t = self._sample_mean_var_val(
                x_t, step_idx, state,
                combined_pad_masks, past_key_values, max_position_ids,
                sample_method, num_steps, compute_values,
            )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self._get_logprob_norm(x_t, x_t_mean, x_t_std)

            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)

        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.rl_config.action_chunk, : self.rl_config.action_env_dim
        ]
        if self.rl_config.joint_logprob:
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

    # ------------------------------------------------------------------
    # Internal: Build prefix+middle KV cache
    # ------------------------------------------------------------------

    def _build_prefix_middle_cache(self, images, img_masks, lang_tokens, lang_masks):
        """Build prefix and middle KV caches for efficient multi-step denoising.

        Returns:
            combined_pad_masks: [B, prefix_len + middle_len]
            past_key_values: KV cache containing prefix + middle
            max_position_ids: [B, 1] max position id for suffix offset
        """
        # Set eager attention for inference
        self.reasoning_spatial_expert.reasoning_expert.language_model.config._attn_implementation = "eager"
        self.reasoning_spatial_expert.spatial_expert.config._attn_implementation = "eager"
        self.reasoning_spatial_expert.action_expert.config._attn_implementation = "eager"

        # 1. Process prefix (images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self.get_position_ids(prefix_pad_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        normalizer = torch.tensor(
            prefix_embs.shape[-1] ** 0.5, dtype=prefix_embs.dtype, device=prefix_embs.device
        )
        prefix_embs_unscaled = prefix_embs / normalizer

        _, past_key_values = self.reasoning_spatial_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_unscaled, None, None],
            use_cache=True,
        )
        max_prefix_pos = prefix_position_ids.max(dim=-1, keepdim=True).values

        # 2. Process middle (spatial features via VGGT)
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(
            images, img_masks
        )
        middle_len = middle_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        device = middle_embs.device

        prefix_pad_2d_masks = middle_pad_masks[:, :, None] & prefix_pad_masks[:, None, :]
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

        middle_position_ids = torch.arange(1, middle_len + 1, dtype=torch.long, device=device)
        middle_position_ids = middle_position_ids.unsqueeze(0).expand(batch_size, -1)
        middle_position_ids = middle_position_ids + max_prefix_pos

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        normalizer_m = torch.tensor(
            middle_embs.shape[-1] ** 0.5, dtype=middle_embs.dtype, device=middle_embs.device
        )
        middle_embs_unscaled = middle_embs / normalizer_m

        (_, middle_out, _), past_key_values = self.reasoning_spatial_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=middle_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, middle_embs_unscaled, None],
            use_cache=True,
        )

        max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
        combined_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)

        return combined_pad_masks, past_key_values, max_position_ids

    # ------------------------------------------------------------------
    # Internal: denoise step using KV cache
    # ------------------------------------------------------------------

    def _denoise_step_cached(
        self, state, curr_pad_masks, past_key_values, max_position_ids, x_t, timestep
    ):
        """Single denoise step reusing prefix+middle KV cache."""
        from transformers.cache_utils import DynamicCache

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )

        if (
            self.reasoning_spatial_expert.reasoning_expert.language_model.model.layers[0]
            .self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = curr_pad_masks.shape[0]
        device = suffix_embs.device

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # Copy KV cache to avoid mutating the shared cache across denoise steps
        past_key_values_copy = DynamicCache()
        past_key_values_copy.key_cache = list(past_key_values.key_cache)
        past_key_values_copy.value_cache = list(past_key_values.value_cache)
        if hasattr(past_key_values, "_seen_tokens"):
            past_key_values_copy._seen_tokens = past_key_values._seen_tokens

        cached_seq_len = past_key_values_copy.get_seq_length()

        if cached_seq_len > 0:
            cached_mask = suffix_pad_masks[:, :, None] & curr_pad_masks[:, None, :cached_seq_len]
            full_att_2d_masks = torch.cat([cached_mask, suffix_att_2d_masks], dim=2)
        else:
            full_att_2d_masks = suffix_att_2d_masks

        position_ids = torch.arange(1, suffix_len + 1, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids + max_position_ids

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        normalizer_s = torch.tensor(
            suffix_embs.shape[-1] ** 0.5, dtype=suffix_embs.dtype, device=suffix_embs.device
        )
        suffix_embs_unscaled = suffix_embs / normalizer_s

        outputs_embeds, _ = self.reasoning_spatial_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values_copy,
            inputs_embeds=[None, None, suffix_embs_unscaled],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t, suffix_out

    # ------------------------------------------------------------------
    # Internal: sample_mean_var_val (mirrors OpenPI pattern)
    # ------------------------------------------------------------------

    def _sample_mean_var_val(
        self, x_t, idx, state,
        curr_pad_masks, past_key_values, max_position_ids,
        sample_method, denoise_steps, compute_values=True,
    ):
        bsize = state.shape[0]
        device = state.device

        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        noise_level = self._get_noise_level(device=device, dtype=x_t.dtype)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])

        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        v_t, suffix_out = self._denoise_step_cached(
            state, curr_pad_masks, past_key_values, max_position_ids,
            x_t, t_input,
        )

        # Value prediction
        if self.rl_config.add_value_head and compute_values and not self.rl_config.value_after_vlm:
            value_t = self._compute_value_from_suffix(suffix_out)
        else:
            value_t = torch.zeros(bsize, device=device)

        # Compute mean and variance
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if sample_method == "flow_ode":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif sample_method == "flow_sde":
            denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigma_ratio = timesteps / (1 - denom_timesteps)
            sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        else:
            raise ValueError(f"Invalid noise method: {sample_method}")

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t, v_t

    def _get_noise_level(self, device, dtype=None, sample_method=None):
        noise_level = self.rl_config.noise_level
        if self.rl_config.noise_anneal:
            start, end, steps = self.rl_config.noise_params
            progress = min(self.global_step / steps, 1.0)
            noise_level = start + (end - start) * progress
        return torch.tensor(noise_level, device=device, dtype=dtype or torch.float32)

    # ------------------------------------------------------------------
    # Internal: velocity with full forward (for training, no KV cache)
    # ------------------------------------------------------------------

    def _get_velocity_full_forward(
        self, images, img_masks, lang_tokens, lang_masks, state, x_t, timestep
    ):
        """Compute velocity using full forward pass (no KV cache)."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(
            images, img_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )

        if (
            self.reasoning_spatial_expert.reasoning_expert.language_model.model.layers[0]
            .self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            middle_embs = middle_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = self.get_position_ids(pad_masks)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (_, middle_out, suffix_out), _ = self.reasoning_spatial_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
            use_cache=False,
        )

        suffix_out = suffix_out[:, -self.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t, suffix_out

    # ------------------------------------------------------------------
    # Log-prob and value computation for RL training pass
    # ------------------------------------------------------------------

    def get_log_prob_value(
        self, images, img_masks, lang_tokens, lang_masks, state,
        chains, denoise_inds, compute_values=False,
    ):
        bsize = state.shape[0]

        combined_pad_masks, past_key_values, max_position_ids = (
            self._build_prefix_middle_cache(images, img_masks, lang_tokens, lang_masks)
        )

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.rl_config.joint_logprob:
            num_steps = self.rl_config.num_steps
            initial_log_prob = self._get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self._gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]

            x_t_mean, x_t_std, value_t, _ = self._sample_mean_var_val(
                chains_pre, denoise_ind, state,
                combined_pad_masks, past_key_values, max_position_ids,
                self.rl_config.noise_method, self.rl_config.num_steps,
                compute_values,
            )

            log_probs = self._get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self._gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            chains_values.append(value_t)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        chains_entropy = torch.zeros_like(chains_log_probs)

        return chains_log_probs, chains_values, chains_entropy

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _get_logprob_norm(self, sample, mu, sigma):
        if self.rl_config.safe_get_logprob:
            return -torch.pow((sample - mu), 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def _gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        return 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))

    def _compute_value_from_suffix(self, suffix_out):
        if self.rl_config.chunk_critic_input:
            suffix_out_value = torch.mean(
                suffix_out[:, : self.rl_config.action_chunk], dim=1, keepdim=False
            )
        else:
            suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
        if self.rl_config.detach_critic_input:
            suffix_out_value = suffix_out_value.detach()
        return self.value_head(suffix_out_value)[:, 0]

    def _extract_obs_from_forward_inputs(self, forward_inputs):
        """Extract observation dict from forward_inputs for re-creating Observation."""
        obs_dict = {}

        if "image" in forward_inputs:
            obs_dict["image"] = forward_inputs["image"]
            obs_dict["image_mask"] = forward_inputs["image_mask"]
        else:
            images = {}
            image_masks = {}
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
                if f"image/{key}" in forward_inputs:
                    images[key] = forward_inputs[f"image/{key}"]
                    image_masks[key] = forward_inputs.get(
                        f"image_mask/{key}",
                        torch.ones(forward_inputs[f"image/{key}"].shape[0], dtype=torch.bool),
                    )
            if images:
                obs_dict["image"] = images
                obs_dict["image_mask"] = image_masks

        obs_dict["state"] = forward_inputs.get("state", forward_inputs.get("observation/state"))

        if "tokenized_prompt" in forward_inputs:
            obs_dict["tokenized_prompt"] = forward_inputs["tokenized_prompt"]
            obs_dict["tokenized_prompt_mask"] = forward_inputs["tokenized_prompt_mask"]

        return obs_dict

    def _tokenize_prompts(self, processed_obs, prompts):
        """Tokenize text prompts and add to obs dict."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        max_len = self.config.max_token_len or 128

        encoded = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        device = processed_obs["state"].device
        processed_obs["tokenized_prompt"] = encoded["input_ids"].to(device)
        processed_obs["tokenized_prompt_mask"] = encoded["attention_mask"].bool().to(device)
        return processed_obs
