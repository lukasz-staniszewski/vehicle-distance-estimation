import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class UserKittiYoloConfig:
    use_dont_care_label: bool
    test_size: float
    val_size: float
    n_epochs: int
    patience: int
    experiment_path: str
    device: str


def read_user_config() -> UserKittiYoloConfig:
    user_config = yaml.load(open(os.path.join(os.path.dirname(__file__), "config.yaml")), Loader=yaml.FullLoader)
    user_config["experiment_path"] = Path(user_config["experiment_path"])
    return UserKittiYoloConfig(**user_config)
