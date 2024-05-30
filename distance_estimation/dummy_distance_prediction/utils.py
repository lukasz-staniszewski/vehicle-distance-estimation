import os
from dataclasses import dataclass

import yaml


@dataclass
class DummyDistancePredictorConfig:
    seed: int


def read_ddp_config() -> DummyDistancePredictorConfig:
    user_config = yaml.load(open(os.path.join(os.path.dirname(__file__), "config.yaml")), Loader=yaml.FullLoader)
    return DummyDistancePredictorConfig(**user_config)
