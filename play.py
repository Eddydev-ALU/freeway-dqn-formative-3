"""Load the trained DQN and watch it play Atari Freeway.

Run with uv, for example:
    uv run python play.py                                         # plays dqn_model.pth in a window
    uv run python play.py --model Keane_Gaju/exp10_model.pth --episodes 3
    uv run python play.py --no-render --episodes 5                # quick score check, no window
    uv run python play.py --video demo/gameplay_demo.mp4          # record one game to a video file
"""

import argparse

import cv2
import torch

# Reuse the exact environment, preprocessing and network from training,
# so the agent sees the game the same way it did while learning.
from train import (
    create_q_network,
    get_device,
    make_env,
    reset_frame_stack,
    select_action,
    stack_to_state,
    step_frame_stack,
)

VIDEO_FPS = 15    # Freeway-v5 repeats each action for 4 frames: 60 fps / 4 = 15 agent steps per second
VIDEO_SCALE = 3   # enlarge the 210x160 screen so the video is easier to watch


def parse_args():
    parser = argparse.ArgumentParser(description="Watch the trained DQN agent play Freeway")
    parser.add_argument("--model", type=str, default="dqn_model.pth", help="path to the saved weights")
    parser.add_argument("--episodes", type=int, default=1, help="number of games to play")
    parser.add_argument("--no-render", action="store_true", help="play without a window (faster)")
    parser.add_argument("--video", type=str, default=None, help="save the game to this .mp4 file instead of a window")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first game")
    return parser.parse_args()


def load_model(model_path, num_actions, device):
    """Rebuild the same CNN as in train.py and load the trained weights into it."""
    q_network = create_q_network(num_actions, device)
    weights = torch.load(model_path, map_location=device)
    q_network.load_state_dict(weights)
    q_network.eval()  # evaluation mode: we only use the network, we do not train it
    return q_network


def main():
    args = parse_args()
    device = get_device()

    # "human" opens a game window in real time; "rgb_array" lets us grab frames for a video
    if args.video:
        render_mode = "rgb_array"
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "human"

    env = make_env(render_mode=render_mode)
    num_actions = env.action_space.n
    q_network = load_model(args.model, num_actions, device)
    print(f"Loaded {args.model} on {device}")

    video_writer = None
    if args.video:
        height, width = 210 * VIDEO_SCALE, 160 * VIDEO_SCALE
        video_writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height))

    scores = []
    for episode in range(1, args.episodes + 1):
        frames = reset_frame_stack(env, seed=args.seed + episode - 1)
        state = stack_to_state(frames)
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Greedy policy: epsilon = 0, so the agent always takes the action with the highest Q-value
            action = select_action(q_network, state, 0.0, num_actions, device)
            frames, reward, terminated, truncated = step_frame_stack(env, frames, action)
            state = stack_to_state(frames)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if video_writer is not None:
                frame = cv2.resize(env.render(), None, fx=VIDEO_SCALE, fy=VIDEO_SCALE,
                                   interpolation=cv2.INTER_NEAREST)
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV expects BGR

        scores.append(episode_reward)
        print(f"Game {episode}: score {episode_reward:.0f} (crossings) in {episode_length} steps")

    env.close()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {args.video}")
    print(f"Average score over {len(scores)} game(s): {sum(scores) / len(scores):.1f}")


if __name__ == "__main__":
    main()
