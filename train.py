"""Train a Deep Q-Network (DQN) agent to play Atari Freeway (ALE/Freeway-v5).

Run with uv, for example:
    uv run python train.py --lr 1e-4 --gamma 0.99 --output-dir Eddy_Irasetsa --exp-name exp01
"""

import argparse
import csv
import os
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


# ===== Environment and preprocessing =====

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


# ===== Experience replay buffer =====

class ReplayBuffer:
    """Stores past experiences (state, action, reward, next_state, done).

    Sampling random batches from the buffer breaks the correlation between
    consecutive frames, which makes training much more stable.
    When the buffer is full, the oldest experience is overwritten.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0  # where the next experience will be written
        self.size = 0      # how many experiences are currently stored

        # Pre-allocate arrays; states are uint8 (0-255) to save memory
        state_shape = (capacity, STACK_SIZE, FRAME_SIZE, FRAME_SIZE)
        self.states = np.zeros(state_shape, dtype=np.uint8)
        self.next_states = np.zeros(state_shape, dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Save one experience, overwriting the oldest one when the buffer is full."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Pick a random batch of experiences and return them as tensors on the device."""
        indices = np.random.randint(0, self.size, size=batch_size)
        states = torch.from_numpy(self.states[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        next_states = torch.from_numpy(self.next_states[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)
        return states, actions, rewards, next_states, dones


# ===== Epsilon-greedy action selection =====

def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Linear epsilon decay.

    Epsilon starts at eps_start and falls in a straight line to eps_end over
    eps_decay steps, then stays at eps_end. Early on the agent explores a lot;
    later it mostly uses what it has learned.
    """
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(q_network, state, epsilon, num_actions, device):
    """Choose an action with the epsilon-greedy rule.

    With probability epsilon  -> random action (exploration).
    With probability 1 - epsilon -> action with the highest Q-value (exploitation).
    """
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)

    # Add a batch dimension: (4, 84, 84) -> (1, 4, 84, 84)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():  # no gradients needed just to pick an action
        q_values = q_network(state_tensor)
    return int(q_values.argmax(dim=1).item())


# ===== Q-network and learning step (to be implemented by Member 2) =====

def create_q_network(num_actions, device):
    """TODO (Member 2): build the CNN with torch.nn and return it on the device.

    Input: a uint8 batch of shape (batch, 4, 84, 84); divide by 255 inside the network.
    Output: one Q-value per action, shape (batch, num_actions).
    """
    raise NotImplementedError("Q-network not implemented yet (Member 2)")


def train_step(online_net, target_net, optimizer, batch, gamma):
    """TODO (Member 2): one DQN update.

    Predict Q-values with the online network, compute Bellman targets with the
    target network, compute the loss, backpropagate and take an optimizer step.
    batch = (states, actions, rewards, next_states, dones) from ReplayBuffer.sample.
    """
    raise NotImplementedError("Training step not implemented yet (Member 2)")


# ===== Main training loop =====

def main():
    args = parse_args()
    device = get_device()
    print(f"Training on {device} with settings: {vars(args)}")

    # Seed everything so experiments can be repeated
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    num_actions = env.action_space.n
    replay_buffer = ReplayBuffer(args.buffer_size)

    # TODO (Member 2): create the online and target networks and the optimizer
    online_net = create_q_network(num_actions, device)
    target_net = create_q_network(num_actions, device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)

    # Log one row per episode to a CSV file, used later for the training plots
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.exp_name}_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "total_steps", "episode_reward", "episode_length", "epsilon"])

    frames = reset_frame_stack(env, seed=args.seed)
    state = stack_to_state(frames)
    episode = 0
    episode_reward = 0.0
    episode_length = 0

    for step in range(1, args.total_steps + 1):
        # 1. Choose an action with epsilon-greedy and play it
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(online_net, state, epsilon, num_actions, device)
        frames, reward, terminated, truncated = step_frame_stack(env, frames, action)
        next_state = stack_to_state(frames)

        # 2. Store the experience. Only "terminated" counts as done for the
        #    Bellman target, because a truncated episode could have continued.
        replay_buffer.add(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward
        episode_length += 1

        # 3. Learn from a random batch once the buffer has enough experiences
        if step > args.learning_starts and step % args.train_freq == 0:
            batch = replay_buffer.sample(args.batch_size, device)
            train_step(online_net, target_net, optimizer, batch, args.gamma)

        # 4. TODO (Member 2): periodically copy the online weights to the target network

        # 5. When the episode ends, start a new one
        if terminated or truncated:
            episode += 1
            log_writer.writerow([episode, step, episode_reward, episode_length, round(epsilon, 4)])
            log_file.flush()  # write to disk now, so the log survives if training stops early
            print(f"episode {episode} | step {step} | reward {episode_reward} "
                  f"| length {episode_length} | epsilon {epsilon:.3f}")
            frames = reset_frame_stack(env)
            state = stack_to_state(frames)
            episode_reward = 0.0
            episode_length = 0

    env.close()
    log_file.close()
    print(f"Training log saved to {log_path}")
    # TODO: save the trained model to dqn_model.pth in args.output_dir


if __name__ == "__main__":
    main()
