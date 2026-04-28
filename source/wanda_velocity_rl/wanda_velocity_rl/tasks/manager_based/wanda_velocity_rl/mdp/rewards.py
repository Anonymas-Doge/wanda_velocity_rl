# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Wanda quadruped locomotion.

Adapted from Boston Dynamics Spot reward suite, tuned for Wanda's:
  - 3 joints per leg: shoulder_pitch, shoulder_roll, tendonDriver
  - Init height: 0.2875m (shorter than Spot)
  - DCMotorCfg: stiffness=150.0, damping=0.1
  - Trot gait pairs: (FL, BR) and (FR, BL)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


# =============================================================================
# TASK REWARDS
# =============================================================================


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time.

    Tuning note for Wanda:
      - Wanda is shorter/lighter than Spot so gait frequency will be higher.
      - Recommended starting mode_time: 0.3 (vs Spot's ~0.5)
      - Lower velocity_threshold (~0.3) since Wanda's max speed is lower.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)

    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)

    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel.

    Tuning note for Wanda:
      - std=0.25 is a reasonable starting point.
      - Increase std if the yaw reward signal vanishes early in training.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    ramp_at_vel: float = 1.0,
    ramp_rate: float = 0.5,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel.

    Tuning note for Wanda:
      - std=0.25 to start; tighten (lower) once basic locomotion is stable.
      - ramp_at_vel: set to ~0.8 since Wanda's top speed is lower than Spot's.
      - ramp_rate: keep at 0.5 initially; increase to push for faster gaits later.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


class WandaGaitReward(ManagerTermBase):
    """Gait enforcing reward for Wanda's trot pattern.

    Enforces a trot gait by synchronizing diagonal foot pairs:
      - Pair 0 (synced): FL + BR
      - Pair 1 (synced): FR + BL

    Wanda joint naming convention assumed:
      - fl_*_joint, fr_*_joint, bl_*_joint, br_*_joint

    Tuning note for Wanda:
      - std: controls sharpness of sync penalty. Start at 0.25.
      - max_err: clip for squared errors. Start at 0.2 (lower than Spot since
        Wanda has shorter stride times).
      - velocity_threshold: ~0.3 m/s recommended for Wanda.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("WandaGaitReward requires exactly two pairs of two feet (trot pattern).")

        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the gait reward.

        Multiplies sync rewards (diagonal pairs in phase) by async rewards
        (diagonal pairs out of phase with each other).
        """
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1

        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3

        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)

        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of a diagonal foot pair."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization between foot pairs."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a target height off the ground.

    Tuning note for Wanda:
      - target_height: Wanda init height is 0.2875m. Foot clearance of ~0.08m
        is a reasonable start (scale down from Spot's ~0.1m proportionally).
      - std: start at 0.025.
      - tanh_mult: start at 2.0; increase to make the reward more sensitive
        to foot velocity during swing.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


# =============================================================================
# SELF-LEVELING REWARDS (priority for Wanda)
# =============================================================================


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation via projected gravity xy components.

    This is the PRIMARY self-leveling signal for Wanda. Weight this heavily
    (e.g. -3.0 to -5.0) especially during flat terrain training.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)


def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity.

    Weighted 0.8 vertical / 0.2 roll+pitch — same as Spot, appropriate for
    Wanda since the dynamics are similar.

    Tuning note: if Wanda oscillates vertically, increase the 0.8 coefficient.
    If it tends to tip sideways, increase the 0.2 coefficient.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


# =============================================================================
# REGULARIZATION PENALTIES
# =============================================================================


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in network action output.

    Important for Wanda's DCMotor actuators — large action deltas cause
    torque spikes that can destabilize the low-damping (0.1) joints.
    Weight recommendation: -0.01 to -0.05.
    """
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in foot air/contact times across all four feet.

    Encourages symmetric gaits. Particularly useful for Wanda since asymmetric
    gaits cause leveling instability.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground.

    Tuning note for Wanda:
      - threshold: 1.0 N is standard; lower to 0.5 N if slip is common on
        smooth terrain (Wanda flat env uses a plane with friction=1.0).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    return torch.sum(is_contact * foot_planar_velocity, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on all 12 joints (3 per leg).

    Tuning note for Wanda:
      - tendonDriver joints are low-stiffness (see commented cfg values).
        This penalty helps prevent oscillation in those joints specifically.
      - Weight recommendation: -2.5e-7 (same as base config dof_acc_l2).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.joint_acc, dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position deviation from default pose.

    Tuning note for Wanda:
      - default_joint_pos must match your URDF's balanced neutral pose.
        Currently all joints init at 0.0 — verify this is actually stable.
      - stand_still_scale: amplifies the penalty when standing still to
        encourage the robot to return to neutral pose between steps.
        Recommended: 2.0–5.0.
      - velocity_threshold: ~0.3 m/s for Wanda.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        reward,
        stand_still_scale * reward,
    )


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques across all 12 joints.

    Tuning note for Wanda:
      - effort_limit_sim is 40 Nm. Keep weight small (-1e-5) to avoid
        over-penalizing the high-stiffness shoulder_roll joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.applied_torque, dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities across all 12 joints.

    Tuning note for Wanda:
      - velocity_limit is 21.0 rad/s. This penalty discourages the policy
        from saturating the motors, especially the tendonDriver joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.joint_vel, dim=1)