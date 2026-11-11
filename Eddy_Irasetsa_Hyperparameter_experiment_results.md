# Hyperparameter Experiment Results

**Member Name:** Eddy Irasetsa

**Focus:** learning rate and gamma (the other four hyperparameters are also varied
so that every experiment is a different combination).

**Fixed settings for all runs:** 500,000 total steps, replay buffer 50,000,
learning starts after 10,000 steps, train every 4 steps, target network update
every 1,000 steps, seed 42. Epsilon decays linearly from ε start to ε end over
"Epsilon decay" steps.

**Baselines:** random agent = 0, always UP = 21.

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 250,000 | Reward stayed at 0 for the first 93 episodes while ε was high (one lucky crossing at episode 72). Learning started at episode 94 (ε ≈ 0.27) and the reward jumped to 20 by episode 102. It then plateaued at about 22 (range 17-26), just above the always-UP baseline of 21, so the agent mainly learned to go up and only sometimes dodges cars. |
| 2 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.05 | 250,000 | Only the learning rate changed vs exp01. Learning started much earlier: steady rewards from episode 71 (ε ≈ 0.45) instead of 94, and the mean reward over episodes 76-100 was 14.2 vs 1.8. Reached 20 by episode 96. The final level was the same as exp01 (about 22, range 16-26), so a higher learning rate made learning faster but did not improve the final score. |
| 3 | 5e-4 | 0.99 | 32 | 1.0 | 0.02 | 250,000 | Best so far. Learning started at episode 70 (ε ≈ 0.44), about the same as exp02, so raising lr beyond 2.5e-4 did not speed learning up further. No instability: in the last 50 episodes the reward never dropped below 20 (range 20-26), and the final level was about 23, roughly 1 point above exp01/exp02. The gain is most likely from the lower ε end (2% random actions instead of 5%), but lr also changed, so the two effects cannot be fully separated. |
| 4 | 1e-3 | 0.99 | 64 | 1.0 | 0.05 | 200,000 | Fastest to reach 20 (episode 86), helped by the faster ε decay (ε was 0.32 at episode 70). It did not collapse, but it peaked around episodes 126-150 (mean 22.6) and then drifted down to about 21 (last-10 mean 21.1, range 19-26). The very high learning rate learned fast but kept shifting the Q-values instead of settling, ending slightly below exp03. |
| 5 | 5e-5 | 0.99 | 32 | 1.0 | 0.10 | 250,000 | Slowest learner: steady rewards only from episode 107 (ε ≈ 0.21). It climbed to 17-19 by episodes 112-117, then collapsed to 0-4 for about 25 episodes (119-143) before recovering and reaching 20 at episode 151. Likely cause: the 50k-step buffer holds only about 24 episodes, so once the policy slipped it filled with failed episodes, and the low learning rate was slow to correct; the 10% random actions helped it find crossings again. It still finished strongly (last-10 mean 22.9, range 20-25) and was still improving at the end. |
| 6 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 250,000 | Only gamma changed vs exp01. Learning started the latest of all runs so far: steady rewards only from episode 113 (ε ≈ 0.12) instead of 94, because with a lower gamma the distant crossing reward looks smaller from the bottom of the road (weaker signal). Once learned, it jumped from 5 to 21 within 6 episodes, and the final level was about the same as exp01 (last-10 mean 23.1, last-50 22.2, range 17-26). Lower gamma slowed learning but did not hurt the final score. |
| 7 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 150,000 | ε reached 0.05 by episode 73 (fast decay), yet steady learning only started at episode 92: even when mostly acting greedily, the very low gamma gave a weak signal at first. After that it learned quickly (20+ by episode 99) and was the most consistent run so far (last-50 mean 22.7, range 19-25, best 27). Smaller, less noisy targets from the low gamma may explain the stability. Freeway mostly needs "go up and avoid nearby cars", which does not require long-term planning. |
| 8 | 1e-4 | 0.999 | 64 | 1.0 | 0.01 | 300,000 | Slowest to reach 20 (episode 166). It got its first steady rewards at episode 103, then stalled at 0-8 for about 50 episodes. With γ 0.999 the horizon is about 1000 steps, so each Q-value sums many future crossings, and the targets are large and noisy, which made learning slow and unsteady (the slow ε decay added to the delay). It was still improving at the end (last-25 mean 23.2, last-10 23.0), so it might have gone further with more steps. |
| 9 | 2.5e-4 | 0.97 | 32 | 0.9 | 0.05 | 200,000 | Fastest learner of all runs: steady rewards from episode 60 (ε ≈ 0.38), 15 by episode 66 and 20 by episode 85. Every setting favoured an early start (lower ε start, faster decay, a fairly high lr, moderate gamma). The final level was the same as most runs (last-50 mean 21.9, range 18-27, best 27), confirming that these settings change the speed of learning more than the final score. |
| 10 | 1e-5 | 0.99 | 64 | 1.0 | 0.05 | 250,000 | Worst run: the reward was 0 in all 244 episodes, not even one lucky crossing. The learning rate was so low that the network barely changed: after training, all three Q-values were still about 0.003, and the greedy action was mostly DOWN (69% of steps). A chicken that mostly presses DOWN never crosses, so no reward ever entered the replay buffer and there was nothing to learn from. |

## Results summary

| Experiment | Mean reward (last 10 episodes) | Best episode reward | Total episodes | Training time |
|---|---|---|---|---|
| 1 | 21.7 | 26 | 244 | 57 min |
| 2 | 21.7 | 26 | 244 | 62 min |
| 3 | 22.5 | 27 | 244 | 61 min |
| 4 | 21.1 | 26 | 244 | 61 min* |
| 5 | 22.9 | 25 | 244 | 55 min* |
| 6 | 23.1 | 26 | 244 | 48 min* |
| 7 | 22.7 | 27 | 244 | 50 min* |
| 8 | 23.0 | 25 | 244 | 65 min* |
| 9 | 22.1 | 27 | 244 | 59 min* |
| 10 | 0.0 | 0 | 244 | 74 min |

\* Experiments 4 and 5 ran at the same time, and so did 6 and 7, and 8 and 9, on the same machine.

## Commands used

Each model (`expNN_model.pth`) and its episode log (`expNN_log.csv`) are saved in the `Eddy_Irasetsa/` folder.

```bash
uv run python train.py --lr 1e-4   --gamma 0.99  --batch-size 32 --eps-start 1.0 --eps-end 0.05 --eps-decay 250000 --output-dir Eddy_Irasetsa --exp-name exp01
uv run python train.py --lr 2.5e-4 --gamma 0.99  --batch-size 32 --eps-start 1.0 --eps-end 0.05 --eps-decay 250000 --output-dir Eddy_Irasetsa --exp-name exp02
uv run python train.py --lr 5e-4   --gamma 0.99  --batch-size 32 --eps-start 1.0 --eps-end 0.02 --eps-decay 250000 --output-dir Eddy_Irasetsa --exp-name exp03
uv run python train.py --lr 1e-3   --gamma 0.99  --batch-size 64 --eps-start 1.0 --eps-end 0.05 --eps-decay 200000 --output-dir Eddy_Irasetsa --exp-name exp04
uv run python train.py --lr 5e-5   --gamma 0.99  --batch-size 32 --eps-start 1.0 --eps-end 0.10 --eps-decay 250000 --output-dir Eddy_Irasetsa --exp-name exp05
uv run python train.py --lr 1e-4   --gamma 0.95  --batch-size 32 --eps-start 1.0 --eps-end 0.05 --eps-decay 250000 --output-dir Eddy_Irasetsa --exp-name exp06
uv run python train.py --lr 1e-4   --gamma 0.90  --batch-size 32 --eps-start 1.0 --eps-end 0.05 --eps-decay 150000 --output-dir Eddy_Irasetsa --exp-name exp07
uv run python train.py --lr 1e-4   --gamma 0.999 --batch-size 64 --eps-start 1.0 --eps-end 0.01 --eps-decay 300000 --output-dir Eddy_Irasetsa --exp-name exp08
uv run python train.py --lr 2.5e-4 --gamma 0.97  --batch-size 32 --eps-start 0.9 --eps-end 0.05 --eps-decay 200000 --output-dir Eddy_Irasetsa --exp-name exp09
uv run python train.py --lr 1e-5   --gamma 0.99  --batch-size 64 --eps-start 1.0 --eps-end 0.05 --eps-decay 250000 --output-dir Eddy_Irasetsa --exp-name exp10
```

## Greedy evaluation of the saved models

The training rewards above include random ε actions. To see what each saved model really learned, I played
each one greedily (ε = 0, as play.py does) for 3 games, plus 10 extra games for the two models that beat the baseline.

| Experiment | Greedy mean reward | Share of UP actions | What the agent does |
|---|---|---|---|
| 1, 2, 3, 4, 6, 7, 9 | 21.0 (every game) | 100% | presses UP on every step: the same as the always-UP baseline |
| 5 | **23.8** (13 games, range 23-26) | 87% | waits for gaps in traffic, beats the baseline in every game |
| 8 | 23.2 (13 games, range 21-26) | 74% | waits for gaps, beats the baseline in most games |
| 10 | 0.0 | 17% | mostly presses DOWN, never crosses |

## Analysis

**Best configuration: experiment 5** (lr 5e-5, γ 0.99, batch 32, ε 1.0 → 0.10 over 250k steps).
It was the slowest learner and even collapsed for about 25 episodes in the middle of training, but it is the only model that
reliably learned to dodge cars: greedy play averaged 23.8 and never scored below 23, while most models
only learned "always press UP" (21).

**Worst configuration: experiment 10** (lr 1e-5, batch 64). The learning rate was too small for the network
to move away from its random starting weights (all Q-values about 0.003). Its greedy action was mostly DOWN, so it
never crossed, never saw a reward, and never learned anything: 0 in all 244 episodes.

**Effect of learning rate:**
- 1e-5 is too low: no learning at all (exp10).
- From 1e-4 to 2.5e-4, learning starts earlier: steady rewards from episode 71 instead of 94 (exp01 vs exp02, where only lr changed).
- Above 2.5e-4 there is no further speed-up (exp03: episode 70). At 1e-3 the score peaked and then drifted down (exp04).
- The final training score was about 22 for every lr that learned. So lr mainly controls how fast learning starts.
- The lowest working lr (5e-5, exp05) learned slowest but produced the best final policy.

**Effect of gamma:**
- Lower gamma delays learning: γ 0.95 started about 19 episodes later than γ 0.99 (exp06 vs exp01, where only γ changed).
  A crossing reward is about 100 steps away, and 0.95^100 ≈ 0.006, so the signal is weak from the bottom of the road.
- Very high gamma (0.999, exp08) made learning slow and unsteady: the agent stalled at 0-8 for about 50 episodes,
  because the targets sum about 1000 steps of future reward and are large and noisy.
- Once learned, γ from 0.90 to 0.99 all reached the same training score (about 22), because Freeway's strategy
  (go up, avoid nearby cars) does not need long-term planning.

**Main lesson:** the training curves made 9 of the 10 runs look almost equal (about 21-23), but greedy evaluation showed
that most agents had settled on the simple "always UP" strategy. The two runs that learned to dodge
(exp05 and exp08) were both slow learners that were still improving at the end and kept exploring longer
(exp05 kept 10% random actions; exp08 decayed ε over 300k steps). Learning fast made the agents lock into
"always UP" early; slower learning with more exploration found the better policy.

**Final choice and justification:** experiment 5's configuration (lr 5e-5, γ 0.99, batch 32,
ε 1.0 → 0.10, decay 250k). It gives the best agent when played greedily, as in play.py (23.8 vs 21 for always-UP),
and it is the only one that consistently dodges cars. Its weakness is slow and unsteady learning (the mid-run collapse),
so with more time I would train it for longer and with a larger replay buffer.
