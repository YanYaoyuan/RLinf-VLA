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
# omni_vla model configs

import dataclasses
import difflib
from typing import Optional

from omni_vla.models_pytorch import omni_config
import omni_vla.training.optimizer as _optimizer
import omni_vla.training.weight_loaders as weight_loaders
from omni_vla.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)


from rlinf.models.embodiment.omni_vla.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)


_CONFIGS = [
    TrainConfig(
        name="omni_libero",
        model=omni_config.OmniConfig(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    """Return a copy of the config with assets/weight paths set from model_path."""
    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, assets_dir=model_path),
        )

    replace_kwargs = {
        "data": data_config,
        "pytorch_weight_path": model_path,
    }
    if dataclasses.is_dataclass(config) and any(
        field.name == "assets_dirs" for field in dataclasses.fields(config)
    ):
        replace_kwargs["assets_dirs"] = model_path

    return dataclasses.replace(config, **replace_kwargs)


def _override_with_data_kwargs(config: TrainConfig, data_kwargs: dict) -> TrainConfig:
    """Return a copy of the config with data_config set from openpi_data."""
    data_config = dataclasses.replace(config.data, **data_kwargs)
    replace_kwargs = {"data": data_config}
    return dataclasses.replace(config, **replace_kwargs)


def get_omni_vla_config(
    config_name: str,
    model_path: Optional[str] = None,
    data_kwargs: Optional[dict] = None,
    batch_size: Optional[int] = None,
) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if data_kwargs is not None:
        config = _override_with_data_kwargs(config, data_kwargs)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    return config
