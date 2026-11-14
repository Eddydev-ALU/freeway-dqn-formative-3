"""Make the training plots from the CSV logs that train.py writes.

Run with uv, for example:
    uv run python plot_training.py --log Keane_Gaju/exp10_log.csv     # the three plots for the best run
    uv run python plot_training.py --compare Ajak_Chol                # one member's 10 experiments together
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

ALWAYS_UP_SCORE = 21     # an agent that only presses UP scores about 21, our baseline to beat
SMOOTHING = 10           # moving average over the last 10 episodes
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29d"


def read_log(log_path):
    """Read one training log into a dict of numpy arrays, one per column."""
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    return {column: np.array([float(row[column]) for row in rows]) for column in rows[0]}


def moving_average(values, window=SMOOTHING):
    """Average of the last `window` values at each point (shorter at the start)."""
    return np.array([values[max(0, i - window + 1): i + 1].mean() for i in range(len(values))])


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_best_run(log_path, out_dir):
    """Save the three plots the assignment asks for: rewards, episode length and steps."""
    log = read_log(log_path)
    episodes = log["episode"]
    name = os.path.splitext(log_path)[0].replace(os.sep, "_")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Rewards, with epsilon underneath (separate panel, same x-axis) to show when exploration dropped
    fig, (ax, ax_eps) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1])
    ax.plot(episodes, log["episode_reward"], color=GRAY, linewidth=1, label="Episode reward")
    ax.plot(episodes, moving_average(log["episode_reward"]), color=BLUE, linewidth=2,
            label=f"{SMOOTHING}-episode average")
    ax.axhline(ALWAYS_UP_SCORE, color=ORANGE, linestyle="--", linewidth=1.5, label="Always-UP baseline (21)")
    style(ax, f"Episode reward during training ({name})", "Reward (crossings)")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    ax_eps.plot(episodes, log["epsilon"], color=BLUE, linewidth=2)
    style(ax_eps, "Exploration rate ε", "ε")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "best_run_rewards.png"), dpi=150)
    plt.close(fig)

    # 2. Episode length (Freeway always ends when the timer runs out, so this is flat at 2048)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, log["episode_length"], color=BLUE, linewidth=2)
    ax.set_ylim(0, log["episode_length"].max() * 1.2)
    style(ax, "Episode length (steps per episode); the game timer fixes it at 2048", "Steps")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "best_run_episode_length.png"), dpi=150)
    plt.close(fig)

    # 3. Total environment steps so far, and the reward against those steps
    fig, (ax_steps, ax_reward) = plt.subplots(1, 2, figsize=(11, 4))
    ax_steps.plot(episodes, log["total_steps"], color=BLUE, linewidth=2)
    style(ax_steps, "Total environment steps", "Steps")
    ax_reward.plot(log["total_steps"], moving_average(log["episode_reward"]), color=BLUE, linewidth=2)
    style(ax_reward, f"Reward ({SMOOTHING}-episode average) vs steps", "Reward (crossings)")
    ax_reward.set_xlabel("Total environment steps")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "best_run_steps.png"), dpi=150)
    plt.close(fig)
    print(f"Saved 3 plots for {log_path} in {out_dir}/")


def plot_member_comparison(member_dir, out_dir):
    """One small chart per experiment, so a member can compare all 10 runs side by side."""
    log_paths = sorted(glob.glob(os.path.join(member_dir, "exp*_log.csv")))
    columns = 5
    rows = int(np.ceil(len(log_paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.6 * rows), sharex=True, sharey=True)

    for ax, log_path in zip(axes.flat, log_paths):
        log = read_log(log_path)
        ax.plot(log["episode"], moving_average(log["episode_reward"]), color=BLUE, linewidth=2)
        ax.axhline(ALWAYS_UP_SCORE, color=ORANGE, linestyle="--", linewidth=1)
        last20 = log["episode_reward"][-20:].mean()
        ax.set_title(f"{os.path.basename(log_path)[:5]}  (last 20: {last20:.1f})", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(log_paths):]:
        ax.axis("off")

    member = os.path.basename(os.path.normpath(member_dir))
    fig.suptitle(f"{member}: {SMOOTHING}-episode average reward per experiment "
                 f"(dashed = always-UP baseline)", x=0.01, ha="left")
    fig.supxlabel("Episode")
    fig.supylabel("Reward (crossings)")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{member}_experiments.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot DQN training logs")
    parser.add_argument("--log", type=str, help="log CSV of the best run (makes the three required plots)")
    parser.add_argument("--compare", type=str, help="member folder whose 10 experiments to compare")
    parser.add_argument("--out-dir", type=str, default="plots", help="folder to save the plots in")
    args = parser.parse_args()

    if args.log:
        plot_best_run(args.log, args.out_dir)
    if args.compare:
        plot_member_comparison(args.compare, args.out_dir)
    if not args.log and not args.compare:
        parser.error("give --log and/or --compare")


if __name__ == "__main__":
    main()
