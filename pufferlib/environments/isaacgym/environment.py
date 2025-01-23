from pdb import set_trace as T

import gymnasium as gym
import numpy as np
import functools
import sys
import os

import isaacgym  # noqa
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import (
    quat_rotate, quat_mul, quat_from_angle_axis, normalize_angle,
    quat_from_euler_xyz, get_axis_params
)

import torch

import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def env_creator(name='ase'):
    return functools.partial(make, name=name)

def make(name, buf=None):
    return HumanoidSMPLX(buf=buf)

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
        env = IsaacEnv(cfg=None, enable_camera_sensors=False)
        self.gym = env.gym
        self.env = env

        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.num_agents = env.num_envs
        self.log_interval = log_interval

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
    def __init__(self, control_freq_inv=2, headless=True, device='cuda',
            physics_engine=gymapi.SIM_PHYSX, graphics_disable_camera_sensors=False):
        self.gym = gymapi.acquire_gym()
        self.control_freq_inv = control_freq_inv
        self.headless = headless
        self.device = device

        compute_device = -1 if 'cuda' not in device else 0
        graphics_device = -1 if (not graphics_disable_camera_sensors and headless) else 0

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
        sim_params.flex.warm_start = 0.25

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        # create envs, sim and viewer
        self.sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

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

    def step(self):
        self.render()
        #for i in range(self.control_freq_inv):
        #    self.gym.simulate(self.sim)

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
    def __init__(self, num_envs=8, control_freq_inv=2, headless=True, device='cuda',
            physics_engine=gymapi.SIM_PHYSX, graphics_disable_camera_sensors=False,
            smplx_capsule_path="resources/smplx_capsule.xml", spacing=5, buf=None
        ):
        self.env = IsaacEnv(control_freq_inv, headless, device,
            gymapi.SIM_PHYSX, graphics_disable_camera_sensors)
        self.viewer = self.env.viewer
        self.gym = self.env.gym
        self.sim = self.env.sim
        self.device = device

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
        self.num_envs = num_envs
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
            start_pose.p = gymapi.Vec3(*get_axis_params(char_h, 2))
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset,
                start_pose, "humanoid", group=env_id, filter=0, segmentationId=0)
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

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=device)

        ### Build PD action offset and scale
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()
        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=device)

        # Feet?
        self.contact_bodies = ["Head"] 
        self.debug_viz = False

        # Initialize the sim now
        self.gym.prepare_sim(self.sim)

        # get gym GPU state tensors
        self._root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        num_actors = self._root_states.shape[0] // self.num_envs
        self._humanoid_root_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0
        self._humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=device, dtype=torch.int32
        )

        self._dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 1]
        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(
            self._dof_vel, device=device, dtype=torch.float)

        #self._sensor_tensor = gymtorch.wrap_tensor(self.gym.acquire_force_sensor_tensor(self.sim))

        self._rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., : self.num_bodies, 10:13]

        self._contact_forces = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        ).view(self.num_envs, bodies_per_env, 3)[..., : self.num_bodies, :]
        self._dof_force_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_dof_force_tensor(self.sim)
        ).view(self.num_envs, self.num_dof)

        self._terminate_buf = torch.ones(self.num_envs, device=device, dtype=torch.long)
        self._termination_heights = 0.3
        self._termination_heights = to_torch(
            self._termination_heights, device=device
        )

        # TODO: Why is this too small?
        '''
        self._target_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 1, :]

        self._tar_actor_ids = (
            to_torch(
                num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32
            )
            + 1
        )
        '''

        key_bodies = ["Head", "L_Knee", "R_Knee", "L_Elbow", "R_Elbow", "L_Ankle", "R_Ankle", "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3","L_Thumb3","R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3","R_Thumb3"] 
        obj_obs_size = 15
        self.ref_hoi_obs_size = 324 + len(key_bodies) * 3
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
            shape=(1 + (52) * (3 + 6 + 3 + 3) - 3 + 10 * 3 + obj_obs_size + self.ref_hoi_obs_size,),
            dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(low=-1, high=1,
                shape=(51*3,), dtype=np.float32)
        self.num_agents = num_envs
        super().__init__(buf=buf)

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        dof_num = self.gym.get_asset_dof_count(humanoid_asset)


        '''
        # Check if body names are valid
        for body_name in key_bodies:
            body_id = self.gym.find_asset_rigid_body_index(humanoid_asset, body_name)
            print(f"Body name: {body_name}, body_id: {body_id}")
            if body_id == -1:
                print(f"Warning: Body '{body_name}' not found in the asset!")
        '''

        # Key body ids
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in key_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1
            body_ids.append(body_id)

        self._key_body_ids = to_torch(body_ids, device=device, dtype=torch.long)

        # Contact body ids
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in self.contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1
            body_ids.append(body_id)

        self._contact_body_ids = to_torch(body_ids, device=device, dtype=torch.long)

        # Buffers
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if self.viewer is None:
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
        #self.gym.refresh_force_sensor_tensor(self.sim)
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
        #tar_states = self._target_states[env_ids]
        #task_obs = compute_obj_observations(root_states, tar_states)

        # Joint observations
        #obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        obs = torch.cat([humanoid_obs], dim=-1)

        #ts = self.progress_buf[env_ids].clone()
        #self._curr_ref_obs[env_ids] = self.hoi_data_dict[0]["hoi_data"][ts].clone()
        #next_ts = torch.clamp(ts + 1, max=self.max_episode_length - 1)
        #ref_obs = self.hoi_data_dict[0]["hoi_data"][next_ts].clone()
        #self.obs_buf[env_ids] = torch.cat((obs, ref_obs), dim=-1)

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
        #self.progress_buf[env_ids] = self.motion_times.clone()
        #self.reset_buf[env_ids] = 0
        #self._terminate_buf[env_ids] = 0
        self._refresh_sim_tensors()
        #self._compute_observations(env_ids)

    def step(self, actions):
        #self.actions = actions.to(self.device).clone()
        #pd_tar_tensor = gymtorch.unwrap_tensor(self._pd_action_offset + self._pd_action_scale*self.actions)
        #self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

        self.env.step()

        #self.progress_buf += 1
        #self._refresh_sim_tensors()

        '''
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
        '''

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

def compute_obj_observations(root_states, tar_states):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = quat_to_tan_norm(local_tar_rot)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
    return obs


#@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone().view(contact_forces.shape[0],-1)
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, body_contact_buf), dim=-1)
    return obs

def compute_humanoid_reward(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, len_keypos, w):
    ## type: (Tensor, Tensor, Tensor, Tensor, Int, float) -> Tensor

    ### data preprocess ###

    # simulated states
    root_pos = hoi_obs[:, :3]
    root_rot = hoi_obs[:, 3 : 3 + 4]
    dof_pos = hoi_obs[:, 7 : 7 + 51 * 3]
    dof_pos_vel = hoi_obs[:, 160 : 160 + 51 * 3]
    obj_pos = hoi_obs[:, 313 : 313 + 3]
    obj_rot = hoi_obs[:, 316 : 316 + 4]
    obj_pos_vel = hoi_obs[:, 320 : 320 + 3]
    key_pos = hoi_obs[:, 323 : 323 + len_keypos * 3]
    contact = hoi_obs[:, -1:]  # fake one
    key_pos = torch.cat((root_pos, key_pos), dim=-1)
    body_rot = torch.cat((root_rot, dof_pos), dim=-1)
    ig = key_pos.view(-1, len_keypos + 1, 3).transpose(0, 1) - obj_pos[:, :3]
    ig = ig.transpose(0, 1).view(-1, (len_keypos + 1) * 3)

    # reference states
    ref_root_pos = hoi_ref[:, :3]
    ref_root_rot = hoi_ref[:, 3 : 3 + 4]
    ref_dof_pos = hoi_ref[:, 7 : 7 + 51 * 3]
    ref_dof_pos_vel = hoi_ref[:, 160 : 160 + 51 * 3]
    ref_obj_pos = hoi_ref[:, 313 : 313 + 3]
    ref_obj_rot = hoi_ref[:, 316 : 316 + 4]
    ref_obj_pos_vel = hoi_ref[:, 320 : 320 + 3]
    ref_key_pos = hoi_ref[:, 323 : 323 + len_keypos * 3]
    ref_obj_contact = hoi_ref[:, -1:]
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos), dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos), dim=-1)
    ref_ig = ref_key_pos.view(-1, len_keypos + 1, 3).transpose(0, 1) - ref_obj_pos[:, :3]
    ref_ig = ref_ig.transpose(0, 1).view(-1, (len_keypos + 1) * 3)

    ### body reward ###

    # body pos reward
    ep = torch.mean((ref_key_pos - key_pos) ** 2, dim=-1)
    rp = torch.exp(-ep * w["p"])

    # body rot reward
    er = torch.mean((ref_body_rot - body_rot) ** 2, dim=-1)
    rr = torch.exp(-er * w["r"])

    # body pos vel reward
    epv = torch.zeros_like(ep)
    rpv = torch.exp(-epv * w["pv"])

    # body rot vel reward
    erv = torch.mean((ref_dof_pos_vel - dof_pos_vel) ** 2, dim=-1)
    rrv = torch.exp(-erv * w["rv"])

    rb = rp * rr * rpv * rrv

    ### object reward ###

    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos) ** 2, dim=-1)
    rop = torch.exp(-eop * w["op"])

    # object rot reward
    eor = torch.zeros_like(ep)  # torch.mean((ref_obj_rot - obj_rot)**2,dim=-1)
    ror = torch.exp(-eor * w["or"])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel) ** 2, dim=-1)
    ropv = torch.exp(-eopv * w["opv"])

    # object rot vel reward
    eorv = torch.zeros_like(ep)  # torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    rorv = torch.exp(-eorv * w["orv"])

    ro = rop * ror * ropv * rorv

    ### interaction graph reward ###

    eig = torch.mean((ref_ig - ig) ** 2, dim=-1)
    rig = torch.exp(-eig * w["ig"])

    ### simplified contact graph reward ###

    # Since Isaac Gym does not yet provide API for detailed collision detection in GPU pipeline,
    # we use force detection to approximate the contact status.
    # In this case we use the CG node istead of the CG edge for imitation.
    # TODO: update the code once collision detection API is available.

    ## body ids
    # Pelvis, 0
    # L_Hip, 1
    # L_Knee, 2
    # L_Ankle, 3
    # L_Toe, 4
    # R_Hip, 5
    # R_Knee, 6
    # R_Ankle, 7
    # R_Toe, 8
    # Torso, 9
    # Spine, 10
    # Chest, 11
    # Neck, 12
    # Head, 13
    # L_Thorax, 14
    # L_Shoulder, 15
    # L_Elbow, 16
    # L_Wrist, 17
    # L_Hand, 18-32
    # R_Thorax, 33
    # R_Shoulder, 34
    # R_Elbow, 35
    # R_Wrist, 36
    # R_Hand, 37-51

    # body contact
    contact_body_ids = [0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35]
    body_contact_buf = contact_buf[:, contact_body_ids, :].clone()
    body_contact = torch.all(torch.abs(body_contact_buf) < 0.1, dim=-1)
    body_contact = torch.all(body_contact, dim=-1).to(
        float
    )  # =1 when no contact happens to the body

    # object contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(
        float
    )  # =1 when contact happens to the object

    ref_body_contact = torch.ones_like(ref_obj_contact)  # no body contact for all time
    ecg1 = torch.abs(body_contact - ref_body_contact[:, 0])
    rcg1 = torch.exp(-ecg1 * w["cg1"])
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:, 0])
    rcg2 = torch.exp(-ecg2 * w["cg2"])

    rcg = rcg1 * rcg2

    ### task-agnostic HOI imitation reward ###
    reward = rb * ro * rig * rcg

    return reward

def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
    hoi_ref,
    hoi_obs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        body_height = rigid_body_pos[:, 0, 2]  # root height
        body_fall = body_height < termination_heights  # [4096]
        has_failed = body_fall.clone()
        has_failed *= progress_buf > 1

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated

def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

#@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

@torch.jit.script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

@torch.jit.script
def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

#@torch.jit.script
def euler_xyz_to_exp_map(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map

#@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis

#@torch.jit.script
def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q

#@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta);
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta);

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta;
    ratioB = torch.sin(t * half_theta) / sin_half_theta; 
    
    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

#@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

#@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

#@torch.jit.script
def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q
