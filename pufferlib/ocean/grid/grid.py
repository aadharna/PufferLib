from pdb import set_trace as T

import numpy as np
import os

import gymnasium

import pufferlib
from pufferlib.ocean.grid.cy_grid import CGrid

from pufferlib.learning_progress import BidirectionalLearningProgess

class Grid(pufferlib.PufferEnv):
    def __init__(self, render_mode='raylib', vision_range=5,
            num_sims=4096, num_maps=1000, map_size=-1, max_map_size=9,
            report_interval=128, buf=None, ema_alpha = 0.001, p_theta = 0.05, num_active_tasks = 16, rand_task_rate = 0.2, 
                 sample_threshold = 15, memory = 25):
        # breakpoint()
        self.obs_size = 2*vision_range + 1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_sims
        self.report_interval = report_interval
        super().__init__(buf=buf)
        self.float_actions = np.zeros_like(self.actions).astype(np.float32)
        # parameters for learning progress
        self.map_seeds = np.linspace(0, 1, num_maps).astype(np.float32)
        self.active_ids = np.zeros(num_sims).astype(np.float32)
        self.uniform_dist = np.ones(num_maps).astype(np.float32) / num_maps
        self.sampling_dist = np.copy(self.uniform_dist)
        self.levels = np.arange(32).astype(np.int32)
        self.c_envs = CGrid(self.observations, self.float_actions, self.map_seeds, self.active_ids,
            self.rewards, self.terminals, num_sims, num_maps, map_size, max_map_size)
        
        self.lp = BidirectionalLearningProgess(num_maps, ema_alpha, p_theta, 
                                               num_active_tasks, rand_task_rate, 
                                               sample_threshold, memory)

        # breakpoint()
        pass

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset(self.levels)
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        self.c_envs.step(self.levels)

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
               info.append(log)

        # if self.eval:
        # catch outcomes
        rollout_done = any(self.terminals)
        reward_of_done = self.rewards[self.terminals]
        done_ids = self.active_ids[self.terminals].astype(int)
        if rollout_done:
            task_result = {f'tasks/{done_ids[i]}': [reward_of_done[i]] for i in range(len(reward_of_done))}
            if info:
                for k, v in task_result.items():
                    info[0][k] = v
            else:
                info.append(task_result)
        
            self.lp.collect_data(task_result)

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self, overlay=0):
        self.c_envs.render(overlay=overlay)

    def close(self):
        self.c_envs.close()

    def notify(self):
        self.sampling_dist, self.levels = self.lp.calculate_dist()
        self.lp_dist = self.sampling_dist

def test_performance(timeout=10, atn_cache=1024):
    env = CGrid(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_envs * tick / (time.time() - start))