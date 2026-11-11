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
| 6 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 250,000 | |
| 7 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 150,000 | |
| 8 | 1e-4 | 0.999 | 64 | 1.0 | 0.01 | 300,000 | |
| 9 | 2.5e-4 | 0.97 | 32 | 0.9 | 0.05 | 200,000 | |
| 10 | 1e-5 | 0.99 | 64 | 1.0 | 0.05 | 250,000 | |

## Results summary

| Experiment | Mean reward (last 10 episodes) | Best episode reward | Total episodes | Training time |
|---|---|---|---|---|
| 1 | 21.7 | 26 | 244 | 57 min |
| 2 | 21.7 | 26 | 244 | 62 min |
| 3 | 22.5 | 27 | 244 | 61 min |
| 4 | 21.1 | 26 | 244 | 61 min* |
| 5 | 22.9 | 25 | 244 | 55 min* |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| 9 | | | | |
| 10 | | | | |

\* Experiments 4 and 5 ran at the same time on the same machine.

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

## Analysis

**Best configuration:** TODO (after the runs)

**Worst configuration:** TODO (after the runs)

**Effect of learning rate:** TODO

**Effect of gamma:** TODO

**Final choice and justification:** TODO
