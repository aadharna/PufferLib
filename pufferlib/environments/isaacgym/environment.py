from pdb import set_trace as T

import gymnasium as gym
import numpy as np
import functools
import yaml
import sys

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

class IsaacSMPLXHumanoid(pufferlib.PufferEnv):
    def __init__(self, log_interval=128, buf=None):
        self.env = IsaacEnv(cfg=None, enable_camera_sensors=False)
        self.gym = self.env.gym

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



# Base class for RL tasks
class IsaacEnv:
    def __init__(self, control_freq_inv, headless, device,
            physics_engine, graphics_disable_camera_sensors):
        self.gym = gymapi.acquire_gym()
        self.control_freq_inv = control_freq_inv
        self.headless = headless
        self.device = device

        graphics_device = -1 if (not graphics_disable_camera_sensors and headless) else device

        # Sim params
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0/60.0
        if device == 'cuda':
            assert torch.cuda.is_available(), "CUDA is not available"

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        sim_params.use_gpu_pipeline = 'cuda' in device
        sim_params.num_client_threads = 0

        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 10.0 

        sim_params.physx.use_gpu = 'cuda' in device
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.physx.num_subscenes = 0

        sim_params.flex.num_inner_iterations = 10
        sim_params.flex.warmstart = 0.25

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        # create envs, sim and viewer
        self.sim = self.gym.create_sim(device, graphics_device, physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        self.gym.prepare_sim(self.sim)
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def step(self, actions):
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

    def render(self, sync_frame_time=False):
        if not self.viewer:
            return

        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # fetch results
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        else:
            self.gym.poll_viewer_events(self.viewer)

class HumanoidSMPLX(pufferlib.PufferEnv):
    def __init__(self, control_freq_inv=2, headless=True, device='cuda',
            physics_engine=gymapi.SIM_PHYSX, graphics_disable_camera_sensors=False,
            smplx_capsule_path="resources/smplx/smplx_capsule.xml", spacing=5
        ):

        self.env = IsaacEnv(control_freq_inv, headless, device,
            gymapi.SIM_PHYSX, graphics_disable_camera_sensors)

        ### Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 1.6
        self.gym.add_ground(self.sim, plane_params)

        ### Load humanoid asset
        asset_root = os.path.dirname(smplx_capsule_path)
        asset_file = os.path.basename(smplx_capsule_path)
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)

        # create force sensors at the feet
        sensor_pose = gymapi.Transform()
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        ### Create envs
        self.envs = []
        self.humanoid_handles = []
        num_envs = self.num_envs
        num_per_row = int(np.sqrt(self.num_envs))
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        max_agg_bodies = self.num_bodies + 2
        max_agg_shapes = self.num_shapes + 2
        for env_id in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            ### Build env
            char_h = 0.89
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*torch_utils.get_axis_params(char_h, 2))
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset,
                start_pose, "humanoid", group=env_id, filter=0, segmentationID=0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

            for j in range(self.num_bodies):
                #gymapi.Vec3(0.54, 0.85, 0.2)
                color = gymapi.Vec3(np.random.rand(), np.random.rand(), np.random.rand())
                self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, color)

            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
            self.humanoid_handles.append(humanoid_handle)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        self.dof_limits_lower = []
        self.dof_limits_upper = []
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = torch_utils.to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = torch_utils.to_torch(self.dof_limits_upper, device=self.device)

        ### Build PD action offset and scale
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()
        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = torch_utils.to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = torch_utils.to_torch(self._pd_action_scale, device=self.device)

        self.contact_bodies = ["right_foot", "left_foot"] 
        self.debug_viz = False

        # get gym GPU state tensors
        self._root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        num_actors = self._root_states.shape[0] // self.num_envs
        self._humanoid_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1]
        )[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0
        self._humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        self._dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 1]
        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(
            self._dof_vel, device=self.device, dtype=torch.float)

        self._sensor_tensor = gymtorch.wrap_tensor(self.gym.acquire_force_sensor_tensor(self.sim))

        self._rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., : self.num_bodies, 10:13]
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs

        self._contact_forces = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        ).view(self.num_envs, bodies_per_env, 3)[..., : self.num_bodies, :]
        self._dof_force_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_dof_force_tensor(self.sim)
        ).view(self.num_envs, self.num_dof)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._termination_heights = 0.3
        self._termination_heights = torch_utils.to_torch(
            self._termination_heights, device=self.device
        )

        key_bodies = ["Head", "L_Knee", "R_Knee", "L_Elbow", "R_Elbow", "L_Ankle", "R_Ankle", "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3","L_Thumb3","R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3","R_Thumb3"] 
        obj_obs_size = 15
        self.ref_hoi_obs_size = 324 + len(key_bodies) * 3
        self.single_observation_space = pufferlib.spaces.Box(
            (1 + (52) * (3 + 6 + 3 + 3) - 3 + 10 * 3 + obj_obs_size + self.ref_hoi_obs_size,),
            dtype=np.float32
        )
        self.single_action_space = pufferlib.spaces.Box((51*3,), dtype=np.float32)

        # Key body ids
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in key_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1
            body_ids.append(body_id)

        self._key_body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)

        # Contact body ids
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1
            body_ids.append(body_id)

        self._contact_body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)

        if viewer is None:
            return

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _compute_observations(self, env_ids=slice(None)):
        # Compute humanoid obs
        body_pos = self._rigid_body_pos[env_ids]
        body_rot = self._rigid_body_rot[env_ids]
        body_vel = self._rigid_body_vel[env_ids]
        body_ang_vel = self._rigid_body_ang_vel[env_ids]
        contact_forces = self._contact_forces[env_ids]
        local_root_obs = False
        root_height_obs = True
        humanoid_obs = compute_humanoid_observations_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            local_root_obs,
            root_height_obs,
            contact_forces,
            self._contact_body_ids,
        )

        # Compute task obs
        root_states = self._humanoid_root_states[env_ids]
        tar_states = self._target_states[env_ids]
        task_obs = compute_obj_observations(root_states, tar_states)

        # Joint observations
        obs = torch.cat([humanoid_obs, task_obs], dim=-1)

        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_dict[0]["hoi_data"][ts].clone()
        next_ts = torch.clamp(ts + 1, max=self.max_episode_length - 1)
        ref_obs = self.hoi_data_dict[0]["hoi_data"][next_ts].clone()
        self.obs_buf[env_ids] = torch.cat((obs, ref_obs), dim=-1)

    def reset(self, env_ids=slice(None)):
        if env_ids != slice(None) and len(env_ids) == 0:
            return

        # Reset actors
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        # Reset env tensors
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.progress_buf[env_ids] = self.motion_times.clone()
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)

    def step(self, actions):
        self.actions = actions.to(self.device).clone()
        pd_tar_tensor = gymtorch.unwrap_tensor(self.pd_action_offset + self.pd_action_scale*self.actions)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

        self.env.step_physics()

        self.progress_buf += 1
        self._refresh_sim_tensors()

        # extra calc of self._curr_hoi_obs_buf, for correct calculate of imitation reward
        self._compute_hoi_observations()

        self._compute_observations()
        self.rew_buf[:] = compute_humanoid_reward(
            self._curr_ref_obs,
            self._curr_obs,
            self._contact_forces,
            self._tar_contact_forces,
            len(self._key_body_ids),
            self.reward_weights,
        )
        enable_early_termination = True
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._rigid_body_pos,
            self.max_episode_length,
            enable_early_termination,
            self._termination_heights,
            self._curr_ref_obs,
            self._curr_obs,
        )
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

        ## Return observations, rewards, resets, ...

    def render(self, sync_frame_time=False):
        if not self.viewer:
            super().render(sync_frame_time)
            return

        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(
            char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2]
        )

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos

        # # # fixed camera
        # new_cam_target = gymapi.Vec3(0, 0.5, 1.0)
        # new_cam_pos = gymapi.Vec3(1, -1, 1.6)
        # self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)
        super().render(sync_frame_time)
