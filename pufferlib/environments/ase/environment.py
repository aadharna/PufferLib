from pdb import set_trace as T

import gymnasium
import functools
import yaml

import isaacgym  # noqa
from isaacgym import gymapi
from isaacgym import gymutil

from ase.env.tasks.humanoid_amp_getup import HumanoidAMPGetup
import torch

import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def env_creator(name='ase'):
    return functools.partial(make, name=name)


def make(env_cfg_file, motion_file,
         physx_num_threads=1, physx_num_subscenes=1, physx_num_client_threads=1,
         sim_timestep=1.0 / 60.0, headless=False,
         device_id=0, use_gpu=True, num_envs=1, buf=None):

    sim_params = gymapi.SimParams()
    sim_params.dt = sim_timestep
    sim_params.use_gpu_pipeline = use_gpu
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.physx.num_threads = physx_num_threads
    sim_params.physx.num_subscenes = physx_num_subscenes
    sim_params.num_client_threads = physx_num_client_threads
    #if "sim" in cfg:
    #    gymutil.parse_sim_config(cfg["sim"], sim_params)


    rl_device = "cpu"
    if use_gpu:
        assert torch.cuda.is_available(), "CUDA is not available"
        rl_device = "cuda:" + str(device_id)

    with open(env_cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    assert "env" in cfg, "env is not set in the config file"
    assert "sim" in cfg, "sim is not set in the config file"

    # Fill in the env config
    cfg["env"]["numEnvs"] = num_envs
    cfg["env"]["motion_file"] = motion_file

    # Use gpu and physx by default
    # NOTE: Start with training low-level controller, HumanoidAMPGetup
    task = HumanoidAMPGetup(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        device_type=rl_device,  # "cuda" if torch.cuda.is_available() and args.cuda else "cpu",
        device_id=device_id,
        headless=headless,
    )

    env = ASEPufferEnv(task, buf=buf)

class ASEPufferEnv(pufferlib.PufferEnv):
    def __init__(self, env, buf=None):
        self.env = env
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.num_agents = env.num_agents
        super().__init__(buf)

    def reset(self, seed=None):
        obs, _ = self.env.reset()
        self.observations[:] = obs
        return self.observations, {}

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        self.observations[:] = obs
        self.rewards[:] = reward
        self.terminals[:] = done
        self.truncations[:] = False
        return self.observations, self.rewards, self.terminals, self.truncations, info
