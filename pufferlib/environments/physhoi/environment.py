from pdb import set_trace as T

import gymnasium as gym
import numpy as np
import functools
import yaml

import isaacgym  # noqa
from isaacgym import gymapi
from isaacgym import gymutil

from physhoi.env.tasks.physhoi import PhysHOI_BallPlay
from physhoi.env.tasks.task_wrappers import VecTaskWrapper
import torch

import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def env_creator(name='ase'):
    return functools.partial(make, name=name)


def make(name, env_cfg_file, motion_file,
         physx_num_threads=1, physx_num_subscenes=1, physx_num_client_threads=1,
         sim_timestep=1.0 / 60.0, headless=True,
         device_id=0, use_gpu=True, num_envs=32, buf=None):

    sim_params = gymapi.SimParams()
    sim_params.dt = sim_timestep
    sim_params.use_gpu_pipeline = use_gpu
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.physx.num_threads = physx_num_threads
    sim_params.physx.num_subscenes = physx_num_subscenes
    sim_params.num_client_threads = physx_num_client_threads

    rl_device = "cpu"
    if use_gpu:
        assert torch.cuda.is_available(), "CUDA is not available"
        rl_device = "cuda:" + str(device_id)

    with open(env_cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    assert "env" in cfg, "env is not set in the config file"
    assert "sim" in cfg, "sim is not set in the config file"

    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Fill in the env config
    cfg["env"]["numEnvs"] = num_envs
    cfg["env"]["motion_file"] = motion_file

    # Patch paths
    cfg["env"]["asset"]["assetRoot"] = 'PhysHOI/' + cfg["env"]["asset"]["assetRoot"]

    task = PhysHOI_BallPlay(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        device_type=rl_device,  # "cuda" if torch.cuda.is_available() and args.cuda else "cpu",
        device_id=device_id,
        headless=headless,
    )

    envs = VecTaskWrapper(task, rl_device, clip_observations=np.inf, clip_actions=1.0)
    print("num_envs: {:d}".format(envs.num_envs))
    print("num_actions: {:d}".format(envs.num_actions))
    print("num_obs: {:d}".format(envs.num_obs))
    print("num_states: {:d}".format(envs.num_states))

    envs = RecordEpisodeStatisticsTorch(envs, torch.device(rl_device))
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.use_gpu = use_gpu
    assert isinstance(
        envs.single_action_space, pufferlib.spaces.Box
    ), "only continuous action space is supported"

    return PhysHOIPufferEnv(envs, buf=buf)

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None
        self.infos = {
            'episode_return': [],
            'episode_length': [],
        }

    def reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        if env_ids is None:
            self.episode_returns = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.returned_episode_returns = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.returned_episode_lengths = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
        else:
            self.infos['episode_return'] += self.episode_returns[env_ids].tolist()
            self.infos['episode_length'] += self.episode_lengths[env_ids].tolist()
            self.episode_returns[env_ids] = 0
            self.episode_lengths[env_ids] = 0

        return obs

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        return (
            observations,
            rewards,
            dones,
            self.infos,
        )

    def mean_and_log(self):
        info = {
            'episode_return': np.mean(self.infos['episode_return']),
            'episode_length': np.mean(self.infos['episode_length']),
        }
        self.infos = {
            'episode_return': [],
            'episode_length': [],
        }
        return [info]

class PhysHOIPufferEnv(pufferlib.PufferEnv):
    def __init__(self, env, log_interval=128, buf=None):
        self.env = env
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.num_agents = env.num_envs
        self.log_interval = log_interval
        super().__init__(buf)

        # WARNING: ONLY works with native vec. Will break in multiprocessing
        device = torch.device("cuda" if env.use_gpu else "cpu")
        self.observations = torch.from_numpy(self.observations).to(device)
        self.actions = torch.from_numpy(self.actions).to(device)
        self.rewards = torch.from_numpy(self.rewards).to(device)
        self.terminals = torch.from_numpy(self.terminals).to(device)
        self.truncations = torch.from_numpy(self.truncations).to(device)

    def reset(self, seed=None):
        obs = self.env.reset()
        self.observations[:] = obs
        return self.observations, []

    def step(self, actions):
        actions_np = actions
        actions = torch.from_numpy(actions).cuda()
        obs, reward, done, info = self.env.step(actions)
        self.observations[:] = obs
        self.rewards[:] = reward
        self.terminals[:] = done
        self.truncations[:] = False

        done_indices = torch.nonzero(done).squeeze(-1)
        if len(done_indices) > 0:
            self.observations[done_indices] = self.env.reset(done_indices)[done_indices]

        if len(info['episode_return']) > self.log_interval:
            info = self.env.mean_and_log()
        else:
            info = []

        return self.observations, self.rewards, self.terminals, self.truncations, info
