from pufferlib.environments.isaacgym.environment import HumanoidSMPLX, IsaacEnv
import torch

def test_base_env():
    env = IsaacEnv(headless=False)
    while True:
        env.step(None)

def test_humanoid():
    env = HumanoidSMPLX(num_envs=32, headless=False)
    env.reset()
    env.render()
    import time
    while True:
        action = torch.from_numpy(1.0*env.action_space.sample())
        obs = env.step(action)
        env.render()

def test_humanoid_perf(timeout=10, num_envs=4096):
    env = HumanoidSMPLX(num_envs=num_envs, headless=True)
    env.reset()

    import time
    start = time.time()
    action = torch.from_numpy(env.action_space.sample())

    steps = 0
    while time.time() - start < timeout:
        env.step(action)
        steps += num_envs

    end = time.time()
    sps = steps / (end - start)
    print(f"Steps: {steps}, SPS: {sps}")


if __name__ == '__main__':
    #test_base_env()
    #test_humanoid()
    test_humanoid_perf()

