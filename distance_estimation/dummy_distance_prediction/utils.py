import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DummyDistancePredictorConfig:
    test_size: float
    seed: int


def read_ddp_config() -> DummyDistancePredictorConfig:
    user_config = yaml.load(open(os.path.join(os.path.dirname(__file__), "config.yaml")), Loader=yaml.FullLoader)
    return DummyDistancePredictorConfig(**user_config)
