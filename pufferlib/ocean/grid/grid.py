import numpy as np
import os

import gymnasium

import pufferlib
from pufferlib.ocean.grid.cy_grid import CGrid


class BiDirectionalLP:
    def __init__(self, num_maps, ema_alpha = 0.1, p_theta = 0.1):
        """
        parser.add_argument("--ema-alpha", type=float, default=0.1,
                            help="smoothing value for ema in claculating learning progress (default: 0.1)")
        parser.add_argument("--p-theta", type=float, default=0.1,
                            help="parameter for reweighing learning progress (default: 0.1)")
        """
        # self.n = n
        self.outcomes = np.zeros(num_maps).astype(np.float32)
        self.ema_alpha = ema_alpha
        self.p_theta = p_theta
        self.ema_tsr = None
        self.p_fast = None
        self.p_slow = None

    def learning_progress(self):
        # calculate the learning progress
        # normalize task success rates with random baseline rates
        random_success_rate = np.random.rand(self.y.shape[0])
        # normalize the success rate
        norm_y = np.maximum(self.outcomes - random_success_rate, np.zeros(random_success_rate.shape)) / (1.0 - random_success_rate)
        # exponential mean average learning progress
        self.ema_tsr = self.outcomes * self.ema_alpha + self.ema_tsr * (1 - self.ema_alpha) if self.ema_tsr is not None else self.outcomes
        self.p_fast  = norm_y        * self.ema_alpha + self.p_fast  * (1 - self.ema_alpha) if self.p_fast is not None else norm_y
        self.p_slow  = self.p_fast   * self.ema_alpha + self.p_slow  * (1 - self.ema_alpha) if self.p_slow is not None else self.p_fast
        # NOTE: weighting to give more focus to tasks with lower success probabilities
        p_fast_reweigh = ((1 - self.p_theta) * self.p_fast) / (self.p_fast + self.p_theta * (1 - 2 * self.p_fast))
        p_slow_reweigh = ((1 - self.p_theta) * self.p_slow) / (self.p_slow + self.p_theta * (1 - 2 * self.p_slow))
        # learning progress is the change in probability to task success rate
        # NOTE: using bidirectional LP
        learning_progress = np.abs(p_fast_reweigh - p_slow_reweigh)
        unweighted_learning_progress = np.abs(self.p_fast - self.p_slow)
        return learning_progress, unweighted_learning_progress, self.p_fast, self.p_slow, p_fast_reweigh, p_slow_reweigh


class PufferGrid(pufferlib.PufferEnv):
    def __init__(self, render_mode='raylib', vision_range=5,
            num_envs=4096, num_maps=1000, max_map_size=9,
            report_interval=128, buf=None):
        self.obs_size = 2*vision_range + 1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        super().__init__(buf=buf)
        self.map_seeds = np.linspace(0, 1, num_maps).astype(np.float32)
        self.float_actions = np.zeros_like(self.actions).astype(np.float32)
        self.lp = BiDirectionalLP(num_maps)
        self.c_envs = CGrid(self.observations, self.float_actions, self.map_seeds, self.lp.outcomes,
            self.rewards, self.terminals, num_envs, num_maps, max_map_size)
        # breakpoint()
        pass

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
               info.append(log)

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
