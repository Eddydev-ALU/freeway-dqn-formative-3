"""Train a Deep Q-Network (DQN) agent to play Atari Freeway (ALE/Freeway-v5).

Run with uv, for example:
    uv run python train.py --lr 1e-4 --gamma 0.99 --output-dir Member1Name --exp-name exp01
"""

import argparse
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch

# Register the Atari environments with Gymnasium
gym.register_envs(ale_py)

ENV_ID = "ALE/Freeway-v5"
FRAME_SIZE = 84   # each frame is resized to 84 x 84 pixels
STACK_SIZE = 4    # the agent sees the last 4 frames so it can tell how the cars move


def parse_args():
    """Read the hyperparameters from the command line so each experiment is easy to run."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Freeway")

    # The six hyperparameters we tune in our experiments
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for future rewards")
    parser.add_argument("--batch-size", type=int, default=32, help="experiences sampled per update")
    parser.add_argument("--eps-start", type=float, default=1.0, help="starting exploration rate")
    parser.add_argument("--eps-end", type=float, default=0.05, help="final exploration rate")
    parser.add_argument("--eps-decay", type=int, default=250_000,
                        help="number of steps over which epsilon falls from eps-start to eps-end")

    # Training settings that we keep the same across experiments
    parser.add_argument("--total-steps", type=int, default=500_000, help="total environment steps")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="replay buffer capacity")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="steps of experience to collect before learning begins")
    parser.add_argument("--train-freq", type=int, default=4, help="update the network every N steps")
    parser.add_argument("--target-update", type=int, default=1_000,
                        help="copy online weights to the target network every N steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # Where to save the model and logs
    parser.add_argument("--output-dir", type=str, default=".", help="folder to save results in")
    parser.add_argument("--exp-name", type=str, default="dqn_model", help="name of this experiment")
    return parser.parse_args()


def get_device():
    """Use a GPU if one is available (CUDA on Colab, MPS on Apple Silicon), otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Environment and preprocessing
# ---------------------------------------------------------------------------

def make_env(render_mode=None):
    """Create the Freeway environment.

    Freeway-v5 already repeats each chosen action for 4 frames (frameskip=4),
    so we do not add another frame skip ourselves.
    The action space has 3 actions: 0 = NOOP, 1 = UP, 2 = DOWN.
    """
    return gym.make(ENV_ID, render_mode=render_mode)


def preprocess_frame(frame):
    """Turn one 210x160 RGB frame into an 84x84 grayscale frame.

    Colour is not needed to play the game, and a smaller image makes the
    network much faster to train.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)  # keep uint8 (0-255) to save memory in the replay buffer


def reset_frame_stack(env, seed=None):
    """Start a new episode and fill the frame stack with copies of the first frame."""
    frame, _ = env.reset(seed=seed)
    first_frame = preprocess_frame(frame)
    frames = deque([first_frame] * STACK_SIZE, maxlen=STACK_SIZE)
    return frames


def step_frame_stack(env, frames, action):
    """Take one action and push the new frame onto the stack (the oldest frame drops out).

    terminated = the game really ended (the timer ran out in Freeway).
    truncated  = the episode was cut off by a step limit.
    """
    frame, reward, terminated, truncated, _ = env.step(action)
    frames.append(preprocess_frame(frame))
    return frames, reward, terminated, truncated


def stack_to_state(frames):
    """Combine the stacked frames into one state array of shape (4, 84, 84)."""
    return np.array(frames, dtype=np.uint8)


def main():
    args = parse_args()
    device = get_device()
    print(f"Training on {device} with settings: {vars(args)}")


if __name__ == "__main__":
    main()
