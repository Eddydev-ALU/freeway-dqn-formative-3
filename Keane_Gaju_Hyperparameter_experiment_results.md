# Hyperparameter Experiment Results

**Member Name:** Keane Gaju
**Environment:** ALE/Freeway-v5 (DQN, PyTorch)
**Focus hyperparameter:** batch size

## Setup

- Every run trains for 500,000 environment steps (244 episodes of 2,048 steps each), seed 42.
- Runs were done on Kaggle (Tesla T4 GPU).
- Settings not listed in the table were left at the `train.py` defaults: replay buffer 50,000, learning_starts 10,000, train_freq 4, target_update 1000 steps.
- "Reward" is the episode reward during training, i.e. while the agent is still taking random actions with probability epsilon.
- **Rolling 10 >= 10** is the first episode where the average reward of the last 10 episodes reached 10 crossings. It shows how early the agent learned.
- **Avg last 20** is the average reward of the last 20 episodes (exact to 0.05, because the rewards are whole numbers). It is the number used to rank the runs, because a single best episode is too noisy (almost every run has a best episode of 26 or 27).

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 2e-4 | 0.98 | 16 | 1.0 | 0.05 | 250k | Reached a 10-episode average of 10 at episode 84. Final average 22.2 (best 26). |
| 2 | 2e-4 | 0.98 | 32 | 1.0 | 0.05 | 250k | Reached 10 at episode 74, the earliest of the 250k-decay runs. Final average 22.05 (best 26). |
| 3 | 2e-4 | 0.98 | 64 | 1.0 | 0.05 | 250k | Reached 10 at episode 116. Final average 21.6 (best 26). |
| 4 | 2e-4 | 0.98 | 128 | 1.0 | 0.05 | 250k | Slowest of the batch-size runs to reach 10 (episode 129). Final average 21.8 (best 26). |
| 5 | 2e-4 | 0.98 | 256 | 1.0 | 0.05 | 250k | Reached 10 at episode 108. Final average 22.05 (best 26). |
| 6 | 5e-4 | 0.99 | 128 | 1.0 | 0.05 | 250k | Higher learning rate. Reached 10 at episode 119. Final average 22.4 (best 27), joint second best. |
| 7 | 5e-5 | 0.99 | 16 | 1.0 | 0.05 | 250k | Very low learning rate, but it still learned. Reached 10 at episode 100. Final average 22.1 (best 26). |
| 8 | 3e-4 | 0.99 | 256 | 1.0 | 0.02 | 300k | Slowest epsilon decay. Reached 10 at episode 125. Final average 22.4 (best 27). |
| 9 | 2e-4 | 0.995 | 64 | 0.8 | 0.05 | 150k | Fastest epsilon decay and the earliest learner (episode 39), but the lowest final average: 20.95 (best 25). |
| 10 | 1e-4 | 0.97 | 128 | 0.8 | 0.10 | 200k | Best run with seed 42. Reached 10 at episode 72. Final average 28.6 (best 31), about 6 crossings above every other run. A repeat with seed 1 averaged only 20.1 (see Analysis), so this score was partly seed luck. |

## Ranking by final score (average of last 20 episodes)

| Rank | Experiment | Avg last 20 | Best episode |
|---|---|---|---|
| 1 | 10 | 28.6 | 31 |
| 2 (tie) | 6 | 22.4 | 27 |
| 2 (tie) | 8 | 22.4 | 27 |
| 4 | 1 | 22.2 | 26 |
| 5 | 7 | 22.1 | 26 |
| 6 (tie) | 2 | 22.05 | 26 |
| 6 (tie) | 5 | 22.05 | 26 |
| 8 | 4 | 21.8 | 26 |
| 9 | 3 | 21.6 | 26 |
| 10 | 9 | 20.95 | 25 |

## Analysis

**Best run: experiment 10** (lr 1e-4, gamma 0.97, batch 128, epsilon 0.8 to 0.10 over 200k steps, seed 42). It averaged 28.6 crossings over its last 20 episodes, against 20.95 to 22.4 for every other run, and it reached a 10-crossing average at episode 72.

**Second-seed check:** the same configuration repeated with seed 1 (`exp10_seed1`) reached a 10-crossing average earlier (episode 43) but finished with an average of 20.1 over its last 20 episodes (best 24), in line with the other runs. So the configuration is not shown to be better, and the high score of the seed-42 run is likely a seed effect. The seed-to-seed gap for this configuration (about 8 crossings) is larger than the gaps between experiments 1 to 9 (20.95 to 22.4), so those differences are within noise. The seed-42 model is still the strongest saved model, which is what matters when choosing the final `dqn_model.pth`.

**Worst configuration: experiment 9** (lr 2e-4, gamma 0.995, batch 64, epsilon 0.8 to 0.05 over 150k steps). It had the lowest final average (20.95). The difference from most other runs is only about 1 crossing, so with one seed per run it is a weak "worst".

**Batch size (experiments 1 to 5, only batch size changes):** final averages were 22.2, 22.05, 21.6, 21.8 and 22.05. This spread is too small to separate from run-to-run noise, so batch size had no clear effect on the final score. Batch 16 and 32 reached an average of 10 a little sooner (episodes 84 and 74) than batch 64 to 256 (episodes 108 to 129), but the trend is not smooth.

**Epsilon decay:** a shorter decay made the agent start earning reward sooner. Decay over 150k steps reached an average of 10 at episode 39, 200k at episode 72, 250k at episodes 74 to 129, and 300k at episode 125. This agrees with the group's baseline observation that reward stayed at zero while epsilon was high and jumped once it dropped. A faster start did not give a better final score: experiment 9 started fastest but finished last.

**Learning rate:** from 5e-5 (experiment 7) to 5e-4 (experiment 6), final averages stayed between 22.1 and 22.4. Learning rate had little effect in this range.

**Gamma and epsilon end:** the seed-42 run of experiment 10 used the lowest gamma (0.97) and the highest final epsilon (0.10), but its seed-1 repeat did not reproduce the high score, so we cannot conclude that either value helps. Scores in this table were reached while the agent still took random actions (10% in experiment 10), so greedy play in `play.py` may score as well or better.

**Final choice for this member:** the experiment 10 model (seed 42) as the strongest saved model, because it has the highest final score. As a hyperparameter configuration, experiment 10 is not clearly better than the others, since the seed-1 repeat scored 20.1. Experiments 1 to 9 finished within 1.5 crossings of each other, which is smaller than the seed-to-seed gap, so none of them can be called better than another. Each table run used one seed (42).

## Files

Models and logs for these runs are in the `Keane/` folder: `exp01_model.pth` to `exp10_model.pth` and the matching `exp01_log.csv` to `exp10_log.csv`. The seed-1 confirmation run described in the Analysis is not one of the 10 experiments in the table. Its numbers come from its console output, and its model file was not kept.
