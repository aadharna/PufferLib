from pdb import set_trace as T

import gymnasium
import functools
from pathlib import Path

import numpy as np
from env.config import IntersectionZooEnvConfig
from env.task_context import PathTaskContext
from env.environment import IntersectionZooEnv
from sumo.constants import REGULAR

import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def env_creator(name='intersection_zoo'):
    return functools.partial(make, name=name)

def make(name, penetration=0.33, temperature_humidity=68_46, render_mode='rgb_array', buf=None):
    tasks = PathTaskContext(
        dir='temp',
        single_approach=True,
        penetration_rate=penetration,
        temperature_humidity=temperature_humidity,
        electric_or_regular=REGULAR,
    )

    env_conf = IntersectionZooEnvConfig(
        task_context=tasks.sample_task(),
        working_dir='temp',
        moves_emissions_models=[temperature_humidity],
        fleet_reward_ratio=1,
        visualize_sumo=False,
    )

    # Create the environment
    env = IntersectionZooEnv({"intersectionzoo_env_config": env_conf})

    return pufferlib.emulation.PettingZooPufferEnv(env=env, buf=buf)
