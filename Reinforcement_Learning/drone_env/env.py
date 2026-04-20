import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math

OUTPUT_PATH = "episode.mp4"
MODEL_PATH = "ppo_drone_delivery.zip"
MAX_EPISODE_STEPS = 1200

class DroneEnv(gym.Env):
    def __init__(self):
        '''
        Initializes the DroneEnv class and declares constants used in the env
        '''
        super().__init__()

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # Environment constants
        self.dt = 1.0 / 60.0
        self.mass = 1.0
        self.I = 0.1
        self.g = 9.81
        self.drone_width = 0.4
        self.max_thrust = 10.0
        self.width = 14.0
        self.height = 10.0
        self.world_screen_width = 720
        self.ui_width = 200
        self.screen_width = self.world_screen_width + self.ui_width
        self.screen_height = 540
        self.scale = self.world_screen_width / self.width
        self.drone_color = (250, 100, 20) # (BGR)
        self.thrust_color = (20, 20, 250) # Red (BGR)
        self.checkpoint_color = (50, 200, 50) # Green (BGR)
        self.battery_base_drain = 1.0 / 10000.0
        self.battery_thrust_drain = 1.0 / 500.0
        self.checkpoint_radius = 0.45
        self.max_steps = MAX_EPISODE_STEPS
        self.ground_thickness = 0.2
        self.drone_half_height = 5.0 / self.scale
        self.drone_ground_y = self.ground_thickness + self.drone_half_height

        self.reset()
    
    def get_action_space(self):
        '''
        The action consists of two continuous values in [-1, 1] representing left and right rotor thrust controls.
        This needs to be defined for the model, and will be scaled to actual thrust values in the step() function.
        '''
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def get_observation_space(self):
        '''
        The observation consists of 11 continuous values.
        The observation() method will return the actual values, 
        but we need to explicitly define the space here for the model ahead of time.
        '''
        return spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        '''
        Resets the environment to an initial state and returns the initial observation.
        '''
        super().reset(seed=seed)

        # Drone state: x, y, vx, vy, angle, angular_vel, battery
        self.pos_x = self.width / 2.0
        self.pos_y = self.height * 0.65
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.angle = 0.0
        self.angular_vel = 0.0
        self.battery = 1.0

        # Checkpoints
        self.checkpoint = None
        self.next_checkpoint_side = 1.0
        self.total_collected = 0
        self.score = 0.0

        self.step_count = 0
        self.drone_ground_collision = False
        self.last_action = [0.0, 0.0]
        self.collected_this_step = 0
        self.last_reward = 0.0
        self.episode_reward = 0.0

        self._spawn_checkpoint()

        return self.observation(), {}

    def observation(self):
        '''
        Returns observation for the current state of the environment.
        '''
        cp_dx, cp_dy = 0.0, 0.0
        cp_dvx, cp_dvy = -self.vel_x, -self.vel_y
        ground_distance = self.pos_y - self.drone_ground_y
        drone_x_relative = self.pos_x - self.width / 2.0

        if self.checkpoint is not None:
            cp_dx = float(self.checkpoint[0] - self.pos_x)
            cp_dy = float(self.checkpoint[1] - self.pos_y)

        return np.array([
            cp_dx, # X position relative to checkpoint
            cp_dy, # Y position relative to checkpoint
            cp_dvx, # X velocity relative to checkpoint
            cp_dvy, # Y velocity relative to checkpoint
            ground_distance, # Distance to ground
            self.vel_x, # X velocity
            self.vel_y, # Y velocity
            self.angle, # Angle
            self.angular_vel, # Angular velocity
            self.battery, # Battery level
            drone_x_relative # X position relative to the world
        ], dtype=np.float32)

    def step(self, action):
        '''
        Performs one step in the environment based on the provided actions.
        '''
        self.step_count += 1

        obs_before = self.observation()

        # Map actions [-1, 1] to thrusts [-max_thrust, max_thrust].
        left_thrust = float(action[0]) * self.max_thrust
        right_thrust = float(action[1]) * self.max_thrust

        # Battery depletion disables thrust; gravity will eventually pull the drone down.
        if self.battery <= 0.0:
            left_thrust = right_thrust = 0.0

        # Save action values for rendering
        self.last_action = [left_thrust, right_thrust]

        total_thrust = left_thrust + right_thrust
        force_x = -total_thrust * math.sin(self.angle)
        force_y = total_thrust * math.cos(self.angle) - self.mass * self.g

        torque = (right_thrust - left_thrust) * (self.drone_width / 2.0)

        self.angular_vel += (torque / self.I) * self.dt
        self.angle += self.angular_vel * self.dt

        self.vel_x += (force_x / self.mass) * self.dt
        self.vel_y += (force_y / self.mass) * self.dt
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        self.drone_ground_collision = False
        if self.pos_y < self.drone_ground_y:
            self.pos_y = self.drone_ground_y
            if self.vel_y < 0.0:
                self.vel_y = 0.0
            self.vel_x = 0.0
            self.drone_ground_collision = True

        thrust_load = (abs(left_thrust) + abs(right_thrust)) / (2.0 * self.max_thrust)
        self.battery -= self.battery_base_drain
        self.battery -= thrust_load * self.battery_thrust_drain
        self.battery = max(0.0, self.battery)

        # Collect checkpoint when within collection radius.
        checkpoints_collected = 0
        if self.checkpoint is not None and math.hypot(
            float(self.checkpoint[0]) - self.pos_x,
            float(self.checkpoint[1]) - self.pos_y,
        ) <= self.checkpoint_radius:
            checkpoints_collected = 1
            self.total_collected += 1
            num_steps = max(1, self.step_count)
            self.score = 1000.0 * self.total_collected + 100000.0 * (self.total_collected / num_steps)
            self.checkpoint = None
        self.collected_this_step = checkpoints_collected
        self._spawn_checkpoint()

        observation = self.observation()

        info = {
            "collected_this_step": self.collected_this_step,
            "total_collected": self.total_collected,
            "score": self.score,
            "collision": self.drone_ground_collision
        }

        reward = self.reward(obs_before, observation, action, info)
        self.last_reward = float(reward)
        self.episode_reward += float(reward)

        terminated = self.drone_ground_collision
        truncated = self.step_count >= self.max_steps

        return observation, reward, terminated, truncated, info

    def reward(self, obs_before, obs_after, action, info):
        '''
        Computes a reward based on the action as well as the observation before and after.
        '''
        cp_dx_after = float(obs_after[0])
        cp_dy_after = float(obs_after[1])
        cp_dvx_after = float(obs_after[2])
        cp_dvy_after = float(obs_after[3])
        battery_after = float(obs_after[9])
        battery_before = float(obs_before[9])
        angle_after = float(obs_after[7])
        ground_distance_after = float(obs_after[4])
        ground_distance_before = float(obs_before[4])
        ground_distance_change = ground_distance_after - ground_distance_before
        collected_this_step = info.get("collected_this_step", 0)
        collision = info.get("collision", False)

        dist_after = math.hypot(cp_dx_after, cp_dy_after)
        # Positive when moving toward the checkpoint.
        closing_speed = -(
            (cp_dx_after * cp_dvx_after + cp_dy_after * cp_dvy_after)
            / (dist_after + 1e-6)
        )

        reward = 0.0

        # Primary objective: gather checkpoints.
        reward += 1000.0 * float(collected_this_step)

        # Shape toward nearest checkpoint using relative velocity.
        reward += 1.0 * closing_speed # Reward for moving toward the checkpoint
        # reward -= 0.1 * dist_after # Small penalty for distance to encourage getting closer (but not too much to avoid local minima).

        # Mild shaping for battery management and stable flight.
        reward -= 0.01 * (battery_before - battery_after) # Battery usage penalty.
        reward -= 0.003 * abs(angle_after) # Penalty for tilting (encourages stable flight).
        if not collision:
            reward += 0.001 # Small reward for staying in the air.
        # reward -= 0.1 * abs(ground_distance_change) # Penalty for changes in altitude (encourages smooth flight).

        if collision and battery_after > 0.0:
            reward -= 2000.0 # Large penalty for crashing into the ground if battery is not depleted
        # if battery_after <= 0.0 and battery_before > 0.0:
        #     reward -= 100.0 # Large penalty for depleting battery (encourages efficient use).

        return reward
    
    def _spawn_checkpoint(self):
        '''
        Helper method to spawn the checkpoint in the environment.
        '''
        if self.checkpoint is not None:
            return
        for _ in range(64):
            side = self.next_checkpoint_side
            drone_x = float(np.clip(self.pos_x, 0.8, self.width - 0.8))
            y = float(self.np_random.uniform(self.ground_thickness + 1.0, self.height - 0.8))

            if side > 0.0:
                x_min = max(drone_x + 1.0, self.width * 0.55, 0.8)
                x_max = self.width - 0.8
            else:
                x_min = 0.8
                x_max = min(drone_x - 1.0, self.width * 0.45, self.width - 0.8)

            if x_min >= x_max:
                x = x_max if side > 0.0 else x_min
            else:
                x = float(self.np_random.uniform(x_min, x_max))

            self.checkpoint = np.array([x, y], dtype=np.float32)
            self.next_checkpoint_side *= -1.0
            return

    def render(self):
        '''
        Renders the state of the environment and returns an image
        '''
        # Create a white background image
        img = np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 255

        panel_x = self.world_screen_width + 25
        panel_top = 50
        panel_width = 110
        bar_height = 18
        bar_gap = 50

        def draw_bar(y, label, value, color):
            value = float(np.clip(value, 0.0, 1.0))
            cv2.rectangle(img, (panel_x, y), (panel_x + panel_width, y + bar_height), (210, 210, 210), 1)
            fill_width = int(panel_width * value)
            if fill_width > 0:
                cv2.rectangle(img, (panel_x, y), (panel_x + fill_width, y + bar_height), color, -1)
            cv2.putText(img, label, (panel_x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Transform functions
        def to_pixels(x, y):
            return int(x * self.scale), int(self.screen_height - y * self.scale)

        # Draw ground strip
        ground_top = to_pixels(0.0, self.ground_thickness)[1]
        cv2.rectangle(img, (0, ground_top), (self.screen_width, self.screen_height), (225, 225, 225), -1)
        cv2.line(img, (0, ground_top), (self.screen_width, ground_top), (170, 170, 170), 2)

        # Draw checkpoints.
        cp_radius_px = int(self.checkpoint_radius * self.scale)
        if self.checkpoint is not None:
            cp_px = to_pixels(float(self.checkpoint[0]), float(self.checkpoint[1]))
            cv2.circle(img, cp_px, cp_radius_px, self.checkpoint_color, -1)
            cv2.circle(img, cp_px, cp_radius_px, (0, 0, 0), 1)

        # Draw Drone
        drone_px = to_pixels(self.pos_x, self.pos_y)

        half_w = int(self.drone_width / 2.0 * self.scale)
        h = 5
        pts = np.array([[-half_w, -h], [half_w, -h], [half_w, h], [-half_w, h]], dtype=np.int32)

        cos_a = math.cos(-self.angle)
        sin_a = math.sin(-self.angle)

        def rotate_local(point):
            rx = point[0] * cos_a - point[1] * sin_a
            ry = point[0] * sin_a + point[1] * cos_a
            return int(rx + drone_px[0]), int(ry + drone_px[1])

        rotated_pts = [rotate_local(p) for p in pts]

        rotated_pts = np.array(rotated_pts, dtype=np.int32)
        cv2.fillPoly(img, [rotated_pts], self.drone_color)

        # Draw thrust lines.
        left_t, right_t = self.last_action
        for thrust, x_pos in ((left_t, -half_w), (right_t, half_w)):
            if abs(thrust) <= 1e-6:
                continue
            sign = 1.0 if thrust > 0.0 else -1.0
            length = 50.0 * (abs(thrust) / self.max_thrust)
            start = np.array([x_pos, h], dtype=np.float32)
            end = np.array([x_pos, h + sign * length], dtype=np.float32)
            cv2.line(img, rotate_local(start), rotate_local(end), self.thrust_color, 2)

        average_thrust = (abs(left_t) + abs(right_t)) / (2.0 * self.max_thrust)

        # Side panel indicators.
        draw_bar(panel_top, "Battery", self.battery, self.checkpoint_color)
        draw_bar(panel_top + bar_gap, "Thrust", average_thrust, self.thrust_color)
        cv2.putText(img, f"Score: {self.score:.0f}", (panel_x - 10, panel_top + bar_gap * 2 + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Checkpoints: {self.total_collected}", (panel_x - 10, panel_top + bar_gap * 3 + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Step: {self.step_count}", (panel_x - 10, panel_top + bar_gap * 4 + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Reward: {self.last_reward:.2f}", (panel_x - 10, panel_top + bar_gap * 5 + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Total Reward: {self.episode_reward:.2f}", (panel_x - 10, panel_top + bar_gap * 6 + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img

    def close(self):
        pass
