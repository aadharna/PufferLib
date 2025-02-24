from pdb import set_trace as T

import numpy as np
import os

import gymnasium

import pufferlib
from pufferlib.ocean.grid.cy_grid import CGrid


class PufferGrid(pufferlib.PufferEnv):
    def __init__(self, render_mode='raylib', vision_range=5,
            num_envs=4096, num_maps=1000, max_map_size=9,
            report_interval=128, buf=None, eval=False):
        self.eval = eval
        self.obs_size = 2*vision_range + 1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        super().__init__(buf=buf)
        self.float_actions = np.zeros_like(self.actions).astype(np.float32)
        # parameters for learning progress
        self.map_seeds = np.linspace(0, 1, num_maps).astype(np.float32)
        self.ema_alpha = 0.001
        self.p_theta = 0.05
        self.outcomes = np.zeros(num_maps).astype(np.float32)
        self.ema_tsr = np.zeros(num_maps)
        self.p_fast = np.zeros(num_maps)
        self.p_slow = np.zeros(num_maps)
        self.active_ids = np.zeros(num_envs).astype(np.float32)
        self.uniform_dist = np.ones(num_maps).astype(np.float32) / num_maps
        self.sampling_dist = np.copy(self.uniform_dist)
        # self.lp = BiDirectionalLP(num_maps)
        self.c_envs = CGrid(self.observations, self.float_actions, self.map_seeds, self.active_ids,
            self.rewards, self.terminals, num_envs, num_maps, max_map_size)
        # breakpoint()
        pass
        # from random import random
        # cdef int i, j, idx
        # cdef double u, cumulative
        # cdef double s = 0.0

        # # (Optional) Check or normalize distribution
        # for j in range(self.num_maps):
        #     s += p[j]
        # if abs(s - 1.0) > 1e-6:
        #     raise ValueError("Distribution p does not sum to 1.0 (sum = %f)" % s)

        # for i in range(self.num_envs):
        #     u = random()
        #     cumulative = 0.0
        #     for idx in range(self.num_maps):
        #         cumulative += p[idx]
        #         if u < cumulative:
        #             break

        #     self.map_idxs[i] = idx
        #     reset(&self.envs[i], i)
        #     set_state(&self.envs[i], &self.levels[idx])

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset(self.sampling_dist)
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        if self.sampling_dist.sum() > 1:
            T()
        self.c_envs.step(self.sampling_dist)
        
        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
               info.append(log)

        if self.eval:
            # catch outcomes
            rollout_done = any(self.terminals)
            reward_of_done = self.rewards[self.terminals]
            done_ids = self.active_ids[self.terminals].astype(int)
            if rollout_done:
                task_result = {done_ids[i]: reward_of_done[i] for i in range(len(reward_of_done))}
                if info:
                    info[0]['tasks'] = task_result
                else:
                    info.append({'tasks': task_result})

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

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
