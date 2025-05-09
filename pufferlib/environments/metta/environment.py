from pdb import set_trace as T
import functools

import numpy as np

import pufferlib
from mettagrid.gym_wrapper import RaylibRendererWrapper

from pufferlib.learning_progress import LPEnvWrapper

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/dr_metta.yaml', render_mode='auto', buf=None, seed=0, num_maps = 64, 
         ema_alpha = 0.001, p_theta = 0.05, num_active_tasks = 16, rand_task_rate = 0.25, 
         sample_threshold = 10, memory = 25, use_lp = True, lp_metric='episode/reward.mean'):
    '''Crafter creation function'''
    env = MettaPuff(config, render_mode, buf, seed)
    if use_lp:
        env = LPEnvWrapper(env=env, 
                           num_tasks=num_maps, 
                           num_active_tasks=num_active_tasks, 
                           ema_alpha=ema_alpha, 
                           p_theta=p_theta, 
                           rand_task_rate=rand_task_rate, 
                           sample_threshold=sample_threshold, 
                           memory=memory, 
                           lp_metric=lp_metric)
    return env 

class MettaPuff(pufferlib.PufferEnv):
    def __init__(self, config, render_mode='human', buf=None, seed=0):
        self.render_mode = render_mode
        import mettagrid.mettagrid_env
        self.env = mettagrid.mettagrid_env.make_env_from_cfg(config, render_mode, buf=buf)

        if render_mode == 'human':
            from mettagrid.gym_wrapper import RaylibRendererWrapper
            self.env = RaylibRendererWrapper(self.env, self.env._env_cfg)

        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.num_agents = self.env.num_agents
        super().__init__(buf)

        #cfg = self.env._env_cfg
        #cfg.eval.env = config_from_path(cfg.eval.env, cfg.eval.env_overrides)
        #from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer
        #self.env._renderer =  MettaGridRaylibRenderer(self.env._c_env, self.env._env_cfg['game'])


    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)

        if all(term) or all(trunc):
            # if self.use_lp:
            #     # alternate possability, send agent/heart.get
            #     metric = self.lp_metric
            #     self.lp.collect_data({f'tasks/{self._env_cfg_idx}': [info[metric]]})
            #     if self.send_lp_metrics:
            #         info[f'{self._env_cfg_idx}/{metric}'] = info[metric]
            #         self.lp.add_stats(info)
            self.reset()
            self.env.should_reset = True
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']
        else:
            info = []

        return obs, rew, term, trunc, [info]

    def reset(self, seed=None):
        # if self.use_lp:
        #     #     levels = self.lp_levels
        #     # else:
        #     levels = self.lp_levels
        #     self._env_cfg_idx = np.random.choice(levels)
        #     self.env._env_cfg = self.cfgs[self._env_cfg_idx]
        #     self.env._reset_env()

        #     self.env._c_env.set_buffers(
        #         self.env.observations,
        #         self.env.terminals,
        #         self.env.truncations,
        #         self.env.rewards)

        #     obs, infos = self.env._c_env.reset()
        #     self.env.should_reset = False
        #     self.tick = 0
        #     return obs, infos
        # else:
        obs, _ = self.env.reset()
        self.tick = 0
        return obs, []

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def notify(self):
        pass