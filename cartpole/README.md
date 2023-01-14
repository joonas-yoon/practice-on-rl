# Cart-Pole-v1 (DQN)

## Overview

![gif](https://raw.githubusercontent.com/joonas-yoon/practice-on-rl/d0c0f96e14bdcf15dc5182efa397d5cdff03a9ff/dqn_v1%2B/screenshots/cartpolev1_770.gif)

Running 20 mins for alive 1335 steps/durations.
Please see details on the file `run.log` under each directories.

<img width="480" src="https://raw.githubusercontent.com/joonas-yoon/practice-on-rl/d0c0f96e14bdcf15dc5182efa397d5cdff03a9ff/dqn_v1%2B/plot/train.png" />

*Dot: Duration
**Dashed line (blue): 30-Means duration
***Dashed line (very thin): the rate of exponential decay of epsilon

Environment from Gym: https://www.gymlibrary.dev/environments/classic_control/cart_pole/

Action (discrete):

- 0: Push cart to the left
- 1: Push cart to the right

## Requirements

```bash
$ pip install -r requirements.txt
```

## Run (CLI)

```
$ cd dqn_tutorial_en
$ python run.py
```

If you want to run with interactive cells, you can convert it from `run.py`
