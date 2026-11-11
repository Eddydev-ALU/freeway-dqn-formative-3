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
| 2 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.05 | 250,000 | |
| 3 | 5e-4 | 0.99 | 32 | 1.0 | 0.02 | 250,000 | |
| 4 | 1e-3 | 0.99 | 64 | 1.0 | 0.05 | 200,000 | |
| 5 | 5e-5 | 0.99 | 32 | 1.0 | 0.10 | 250,000 | |
| 6 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 250,000 | |
| 7 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 150,000 | |
| 8 | 1e-4 | 0.999 | 64 | 1.0 | 0.01 | 300,000 | |
| 9 | 2.5e-4 | 0.97 | 32 | 0.9 | 0.05 | 200,000 | |
| 10 | 1e-5 | 0.99 | 64 | 1.0 | 0.05 | 250,000 | |

## Results summary

| Experiment | Mean reward (last 10 episodes) | Best episode reward | Total episodes | Training time |
|---|---|---|---|---|
| 1 | 21.7 | 26 | 244 | 29 min |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| 9 | | | | |
| 10 | | | | |

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
