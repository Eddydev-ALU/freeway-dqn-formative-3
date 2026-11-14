# Hyperparameter Experiment Results

**Member Name:** Ajak Chol

**Focus:** ε start, ε end and ε decay (the exploration schedule). Batch size is set to 64 in most runs
as the second changed value, because Keane's runs 1-5 showed batch size had no clear effect on the score.

**Fixed settings for all runs:** 500,000 total steps, replay buffer 50,000, learning starts after 10,000
steps, train every 4 steps, target network update every 1,000 steps, seed 42.

**Question I am testing:** greedy evaluation of the 20 models from Eddy and Keane showed that 17 of them
only learned "always press UP" (score 21). The three that learned to dodge cars all kept exploring longer
(ε end 0.10, or a slow 300k decay). Does more exploration late in training stop the agent from locking
into "always UP"?

**Baselines:** random agent = 0, always UP = 21. Control = the group baseline (Eddy exp01: lr 1e-4,
γ 0.99, batch 32, ε 1.0 → 0.05 over 250k), greedy score 21.0.

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 64 | 1.0 | 0.10 | 250,000 | TODO |
| 2 | 1e-4 | 0.99 | 64 | 1.0 | 0.20 | 250,000 | TODO |
| 3 | 1e-4 | 0.99 | 64 | 1.0 | 0.30 | 250,000 | TODO |
| 4 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 250,000 | TODO |
| 5 | 1e-4 | 0.99 | 64 | 1.0 | 0.10 | 100,000 | TODO |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.10 | 400,000 | TODO |
| 7 | 1e-4 | 0.99 | 64 | 0.5 | 0.10 | 250,000 | TODO |
| 8 | 1e-4 | 0.99 | 64 | 0.1 | 0.01 | 50,000 | TODO |
| 9 | 5e-5 | 0.99 | 32 | 1.0 | 0.20 | 250,000 | TODO |
| 10 | 1e-4 | 0.97 | 128 | 0.8 | 0.10 | 250,000 | TODO |

What each run compares:
- **1-4 (ε end 0.10 / 0.20 / 0.30 / 0.01):** the main question, with everything else equal.
- **5, 1, 6 (decay 100k / 250k / 400k at ε end 0.10):** how long exploration lasts.
- **7, 1 (ε start 0.5 vs 1.0):** whether starting with less random play speeds learning up.
- **8 (ε 0.1 → 0.01 over 50k):** almost no exploration. Expected worst: Eddy exp10 showed an untrained network
  mostly presses DOWN, so without random actions the agent may never cross and never see a reward.
- **9:** Eddy's best run (exp05) with ε end raised from 0.10 to 0.20.
- **10:** Keane's best run (exp10) with decay 250k instead of 200k, to check whether its 30.7 holds up.

## Results summary

Greedy score = `uv run python play.py --model Ajak_Chol/expNN_model.pth --no-render --episodes 3`.

| Experiment | Mean reward (last 20 training episodes) | Greedy score (3 games) | Behaviour when greedy |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

## Analysis

TODO after the runs: best and worst configuration, the effect of ε end, ε decay and ε start, and the final choice.

## Commands used

Each model (`expNN_model.pth`) and its episode log (`expNN_log.csv`) are saved in the `Ajak_Chol/` folder.

```bash
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 1.0 --eps-end 0.10 --eps-decay 250000 --output-dir Ajak_Chol --exp-name exp01
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 1.0 --eps-end 0.20 --eps-decay 250000 --output-dir Ajak_Chol --exp-name exp02
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 1.0 --eps-end 0.30 --eps-decay 250000 --output-dir Ajak_Chol --exp-name exp03
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 1.0 --eps-end 0.01 --eps-decay 250000 --output-dir Ajak_Chol --exp-name exp04
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 1.0 --eps-end 0.10 --eps-decay 100000 --output-dir Ajak_Chol --exp-name exp05
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 1.0 --eps-end 0.10 --eps-decay 400000 --output-dir Ajak_Chol --exp-name exp06
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 0.5 --eps-end 0.10 --eps-decay 250000 --output-dir Ajak_Chol --exp-name exp07
uv run python train.py --lr 1e-4 --gamma 0.99 --batch-size 64  --eps-start 0.1 --eps-end 0.01 --eps-decay 50000  --output-dir Ajak_Chol --exp-name exp08
uv run python train.py --lr 5e-5 --gamma 0.99 --batch-size 32  --eps-start 1.0 --eps-end 0.20 --eps-decay 250000 --output-dir Ajak_Chol --exp-name exp09
uv run python train.py --lr 1e-4 --gamma 0.97 --batch-size 128 --eps-start 0.8 --eps-end 0.10 --eps-decay 250000 --output-dir Ajak_Chol --exp-name exp10
```
