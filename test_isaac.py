from pufferlib.environments.isaacgym.environment import HumanoidSMPLX, IsaacEnv
import torch

def test_base_env():
    env = IsaacEnv(headless=False)
    while True:
        env.step(None)

def test_humanoid():
    env = HumanoidSMPLX(headless=False)
    env.reset()
    for i in range(10):
        action = torch.from_numpy(env.action_space.sample())
        env.step(action)
        env.render()
    breakpoint()
    pass

if __name__ == '__main__':
    #test_base_env()
    test_humanoid()

