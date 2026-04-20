from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import math
import numpy as np
from stable_baselines3 import PPO

import drone_env  # Registers DroneDelivery-v0
from drone_env.env import OUTPUT_PATH


def _generate_model_colors(num_models):
	"""
	Creates visually distinct BGR colors (one per model).
	"""
	if num_models <= 0:
		return []

	colors = []
	for i in range(num_models):
		hue = int((180 * i) / max(1, num_models))
		hsv = np.array([[[hue, 120, 190]]], dtype=np.uint8)
		bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
		colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
	return colors


def _render_composite_frame(
	envs,
	model_colors,
	model_paths,
	cumulative_scores,
	cumulative_checkpoints,
	current_scores,
	current_collected,
	episode_label,
):
	"""
	Renders all environments composited onto a single frame.
	All checkpoints are drawn first (behind), then all drones on top.
	Each checkpoint and drone is drawn in the model's color.
	Scores update in real-time as episodes progress.
	"""
	env = envs[0].unwrapped
	canvas = np.ones((env.screen_height, env.screen_width, 3), dtype=np.uint8) * 255

	# Ground (world view only)
	ground_top = int(env.screen_height - env.ground_thickness * env.scale)
	cv2.rectangle(canvas, (0, ground_top), (env.screen_width, env.screen_height), (225, 225, 225), -1)
	cv2.line(canvas, (0, ground_top), (env.screen_width, ground_top), (170, 170, 170), 2)

	# Episode indicator in top-left corner.
	cv2.putText(
		canvas,
		episode_label,
		(10, 24),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(0, 0, 0),
		1,
		cv2.LINE_AA,
	)

	# Helper to convert world coords to pixel coords
	def to_pixels(x, y):
		return int(x * env.scale), int(env.screen_height - y * env.scale)

	# First pass: render all checkpoints behind
	for model_idx, env_wrapped in enumerate(envs):
		env_unwrapped = env_wrapped.unwrapped
		color = model_colors[model_idx]

		# Draw checkpoint in model's color
		cp_radius_px = int(env_unwrapped.checkpoint_radius * env_unwrapped.scale)
		if env_unwrapped.checkpoint is not None:
			cp_px = to_pixels(float(env_unwrapped.checkpoint[0]), float(env_unwrapped.checkpoint[1]))
			cv2.circle(canvas, cp_px, cp_radius_px, color, -1)

	# Second pass: render all drones on top
	for model_idx, env_wrapped in enumerate(envs):
		env_unwrapped = env_wrapped.unwrapped
		color = model_colors[model_idx]

		# Draw drone as rotated rectangle in model's color
		drone_px = to_pixels(env_unwrapped.pos_x, env_unwrapped.pos_y)
		half_w = int(env_unwrapped.drone_width / 2.0 * env_unwrapped.scale)
		h = 5
		pts = np.array([[-half_w, -h], [half_w, -h], [half_w, h], [-half_w, h]], dtype=np.int32)

		cos_a = math.cos(-env_unwrapped.angle)
		sin_a = math.sin(-env_unwrapped.angle)

		def rotate_local(point):
			rx = point[0] * cos_a - point[1] * sin_a
			ry = point[0] * sin_a + point[1] * cos_a
			return int(rx + drone_px[0]), int(ry + drone_px[1])

		rotated_pts = np.array([rotate_local(p) for p in pts], dtype=np.int32)
		cv2.fillPoly(canvas, [rotated_pts], color)

		# Draw thrust lines
		left_t, right_t = env_unwrapped.last_action
		thrust_color = (20, 20, 250)  # BGR red
		for thrust, x_pos in ((left_t, -half_w), (right_t, half_w)):
			if abs(thrust) <= 1e-6:
				continue
			sign = 1.0 if thrust > 0.0 else -1.0
			length = 50.0 * (abs(thrust) / env_unwrapped.max_thrust)
			start = np.array([x_pos, h], dtype=np.float32)
			end = np.array([x_pos, h + sign * length], dtype=np.float32)
			cv2.line(canvas, rotate_local(start), rotate_local(end), thrust_color, 2)

	# Side panel with leaderboard
	panel_x0 = env.world_screen_width
	cv2.putText(canvas, "Leaderboard", (panel_x0 + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

	# Build ranked list from cumulative + current scores
	num_models = len(model_paths)
	ranked = sorted(
		[(i, float(cumulative_scores[i] + current_scores[i])) for i in range(num_models)],
		key=lambda item: item[1],
		reverse=True,
	)[:10]

	row_y = 55
	for rank, (model_idx_rank, score_rank) in enumerate(ranked, start=1):
		color_rank = model_colors[model_idx_rank]
		checkpoints = int(cumulative_checkpoints[model_idx_rank] + current_collected[model_idx_rank])
		model_name = model_paths[model_idx_rank].stem
		if len(model_name) > 10:
			model_name = model_name[:10] + ".."

		cv2.rectangle(canvas, (panel_x0 + 10, row_y - 10), (panel_x0 + 24, row_y + 4), color_rank, -1)
		cv2.rectangle(canvas, (panel_x0 + 10, row_y - 10), (panel_x0 + 24, row_y + 4), (0, 0, 0), 1)
		cv2.putText(
			canvas,
			f"{rank:>2}. {model_name}",
			(panel_x0 + 30, row_y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 0, 0),
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			canvas,
			f"{score_rank:.0f} ({checkpoints})",
			(panel_x0 + 30, row_y + 16),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.42,
			(40, 40, 40),
			1,
			cv2.LINE_AA,
		)
		row_y += 52

	return canvas


def _discover_model_paths(models_dir):
	"""
	Discovers model checkpoints inside a directory.
	"""
	model_dir_path = Path(models_dir)
	if not model_dir_path.exists() or not model_dir_path.is_dir():
		raise FileNotFoundError(f"models directory not found: {model_dir_path.resolve()}")

	model_paths = sorted(p for p in model_dir_path.iterdir() if p.is_file() and p.suffix == ".zip")
	if not model_paths:
		raise FileNotFoundError(
			f"No .zip model files found in {model_dir_path.resolve()}"
		)

	return model_paths


def eval_many(num_episodes=10, models_dir="models", output_path=None, seed_base=0, render=True, encode_preset="ultrafast", crf=35):
	"""
	Evaluates all PPO model checkpoints found in a models directory,
	then optionally writes a single video with simple frame rendering.

	Each model runs `num_episodes` episodes. In each episode round, all models
	use the same reset seed so they are compared on identical scenarios.

	Args:
		num_episodes: Number of episodes per model.
		models_dir: Directory containing .zip model files.
		output_path: Output video file path. If None, auto-generated.
		seed_base: Random seed base.
		render: Whether to render video output.
		encode_preset: FFmpeg preset for speed/quality (ultrafast, fast, medium, slow). Default: ultrafast.
		crf: Constant rate factor for h264 (0-51, lower=better quality). Default: 35.
	"""
	if num_episodes <= 0:
		raise ValueError("num_episodes must be >= 1")
	if seed_base < 0:
		raise ValueError("seed_base must be >= 0")
	if not isinstance(render, bool):
		raise TypeError("render must be a bool")
	if encode_preset not in ("ultrafast", "fast", "medium", "slow"):
		raise ValueError("encode_preset must be one of: ultrafast, fast, medium, slow")
	if not (0 <= crf <= 51):
		raise ValueError("crf must be between 0 and 51")

	model_paths = _discover_model_paths(models_dir)
	num_models = len(model_paths)

	total_episodes = num_models * num_episodes

	print(f"Evaluating {num_models} model(s):")

	models = [PPO.load(str(model_path)) for model_path in model_paths]

	output_file = None
	if render:
		if output_path is None:
			out = Path(OUTPUT_PATH)
			output_file = out.with_name(f"{out.stem}_many.mp4")
		else:
			output_file = Path(output_path)
		output_file.parent.mkdir(parents=True, exist_ok=True)

	model_colors = _generate_model_colors(num_models)
	video_writer = None
	rendered_frame_count = 0
	episode_stats = []
	cumulative_scores = [0.0] * num_models
	cumulative_checkpoints = [0] * num_models

	episodes_done = 0
	round_idx = 0
	fps = 30

	envs = [gym.make("DroneDelivery-v0") for _ in range(num_models)]
	template_env = envs[0].unwrapped
	sim_dt = float(getattr(template_env, "dt", 0.0))
	sim_fps = int(round(1.0 / sim_dt)) if sim_dt > 0.0 else fps
	if sim_fps <= 0:
		sim_fps = fps
	
	if render:
		print(f"Rendering at {fps} fps from simulation rate {sim_fps} fps.")
		print(f"Encoder preset: {encode_preset}, CRF: {crf}")
		video_writer = imageio.get_writer(
			str(output_file.resolve()),
			fps=fps,
			macro_block_size=1,
			codec="libx264",
			pixelformat="yuv420p",
			output_params=["-preset", encode_preset, "-crf", str(crf)],
		)

	try:
		for episode_idx in range(num_episodes):
			round_idx += 1
			round_seed = seed_base + episode_idx

			observations = []
			terminated = [False] * num_models
			truncated = [False] * num_models
			done = [False] * num_models

			rewards = [0.0] * num_models
			steps = [0] * num_models
			collected = [0] * num_models
			scores = [0.0] * num_models

			# Track per-env recurrent state from policy to avoid unintended cross-env coupling.
			policy_states = [None] * num_models
			episode_starts = [True] * num_models
			render_budget = 0

			# Reset all environments.
			for i, env in enumerate(envs):
				obs, _ = env.reset(seed=round_seed)
				observations.append(obs)

			# Render initial composite frame after all resets.
			if render:
				frame = _render_composite_frame(
					envs,
					model_colors,
					model_paths,
					cumulative_scores,
					cumulative_checkpoints,
					scores,
					collected,
					f"Episode {episode_idx + 1}/{num_episodes}",
				)
				video_writer.append_data(frame)
				rendered_frame_count += 1

			# Step all envs in lockstep, rendering all on one frame.
			while not all(done):
				capture_this_step = False
				if render:
					render_budget += fps
					if render_budget >= sim_fps:
						capture_this_step = True
						render_budget -= sim_fps

				for i, env in enumerate(envs):
					if done[i]:
						continue

					action, next_state = models[i].predict(
						observations[i],
						state=policy_states[i],
						episode_start=episode_starts[i],
						deterministic=True,
					)
					policy_states[i] = next_state
					episode_starts[i] = False

					obs, reward, term, trunc, info = env.step(action)
					observations[i] = obs

					rewards[i] += float(reward)
					steps[i] += 1
					collected[i] = info.get("total_collected", collected[i])
					scores[i] = float(info.get("score", scores[i]))

					terminated[i] = bool(term)
					truncated[i] = bool(trunc)
					done[i] = terminated[i] or truncated[i]
					if done[i]:
						episode_starts[i] = True
						policy_states[i] = None

				# Render composite frame if capture tick
				if render and capture_this_step:
					frame = _render_composite_frame(
						envs,
						model_colors,
						model_paths,
						cumulative_scores,
						cumulative_checkpoints,
						scores,
						collected,
						f"Episode {episode_idx + 1}/{num_episodes}",
					)
					video_writer.append_data(frame)
					rendered_frame_count += 1

			for i in range(num_models):
				episode_stats.append(
					{
						"episode": episodes_done + i + 1,
						"model_index": i,
						"model_name": model_paths[i].name,
						"episode_in_model": episode_idx + 1,
						"seed": round_seed,
						"steps": steps[i],
						"reward": rewards[i],
						"collected": collected[i],
						"score": scores[i],
						"terminated": terminated[i],
						"truncated": truncated[i],
					}
				)
				cumulative_scores[i] += float(scores[i])
				cumulative_checkpoints[i] += int(collected[i])

			episodes_done += num_models
			print(
				f"Finished round {round_idx}/{num_episodes} (seed={round_seed}): +{num_models} episode(s), "
				f"total done={episodes_done}/{total_episodes}"
			)
	finally:
		if video_writer is not None:
			video_writer.close()
		for env in envs:
			env.close()

	if render:
		if rendered_frame_count == 0:
			raise RuntimeError("No frames were generated for eval_many output.")

	print("eval_many complete.")
	if render:
		print(f"Saved overlay render to: {output_file.resolve()}")
	else:
		print("Render disabled; skipped video creation.")
	print("Episode summaries:")
	for stat in episode_stats:
		end_reason = "terminated" if stat["terminated"] else "truncated"

	model_totals = []
	for i, model_path in enumerate(model_paths):
		model_scores = [
			float(stat["score"])
			for stat in episode_stats
			if stat["model_index"] == i
		]
		model_checkpoints = [
			int(stat["collected"])
			for stat in episode_stats
			if stat["model_index"] == i
		]
		total_score = float(sum(model_scores)) if model_scores else 0.0
		total_checkpoints = int(sum(model_checkpoints)) if model_checkpoints else 0
		model_totals.append((model_path.name, total_score, total_checkpoints))

	model_totals.sort(key=lambda item: item[1], reverse=True)

	print("Total score and checkpoints ranking:")
	for rank, (model_name, total_score, total_checkpoints) in enumerate(model_totals, start=1):
		print(f"  {rank:>2}. {model_name}: score={total_score:.2f}, checkpoints={total_checkpoints}")

