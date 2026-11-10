"""Train a Deep Q-Network (DQN) agent to play Atari Freeway (ALE/Freeway-v5).

Run with uv, for example:
    uv run python train.py --lr 1e-4 --gamma 0.99 --output-dir Member1Name --exp-name exp01
"""

import argparse

import torch


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


def main():
    args = parse_args()
    device = get_device()
    print(f"Training on {device} with settings: {vars(args)}")


if __name__ == "__main__":
    main()
