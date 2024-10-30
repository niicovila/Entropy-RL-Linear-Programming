# [Entropy-RL-Linear-Programming](https://repositori.upf.edu/handle/10230/68216)

## About this work
In this work, we aim to investigate the applicability of regularized LP-based RL algorithms. Our objectives include developing a practical implementation of Q-REPS using deep neural networks (DNNs), and understanding the impact of various practical design choices on the algorithm’s final performance through a large-scale empirical study using a slurm cluster. Additionally, we explore a novel primal-dual policy iteration algorithm.

## Requirements
```bash
python>=3.11
poetry==1.1.1
```
### Usage
```bash
poetry install
poetry run python main/algorithms/qreps/classic/elbe/qreps_elbe.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000

tensorboard --logdir runs
```

To use experiment tracking with wandb, run
```bash
wandb login # only required for the first time
poetry run python main/algorithms/qreps/classic/elbe/qreps_elbe.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000 \
    --track \
    --wandb-project-name qreps_test
```
or if using a config file, 
```bash
poetry run python algos/qreps/qreps_main.py \
    --config <path-to-config.json> \
    --track
```

## Q-REPS

### Algorithm definition
<div align="center">
    <img src="assets/img/minmax_qreps.png" width="412" alt="Title 1">
    <img src="assets/img/elbe_qreps.png" width="412" alt="Title 2">
</div>

#### Episodic Reward Curves
![Reference Image](assets/img/comparison.png)

### Gameplay videos

<table>
  <tr>
    <td>
      <img src="https://github.com/niicovila/Entropy-RL-Linear-Programming/assets/76247144/a908b048-765a-4ef8-a833-30dc54ce48d4" alt="Video 1" style="width:100%">
    </td>
    <td>
      <img src="https://github.com/niicovila/Entropy-RL-Linear-Programming/assets/76247144/420a77f6-1f64-4c2c-b998-94b71fef4d3b" alt="Video 2" style="width:100%">
    </td>
    <td>
      <img src="https://github.com/niicovila/Entropy-RL-Linear-Programming/assets/76247144/87b9ceaf-9bfa-4914-9e12-945478d5066c" alt="Video 3" style="width:100%">
    </td>
  </tr>
</table>

### Continuous Environments extension
- HalfCheetah-v4
<img width="1390" alt="Screenshot 2024-06-21 at 14 17 23" src="https://github.com/niicovila/Entropy-RL-Linear-Programming/assets/76247144/da332a9e-3f5b-4979-a5c7-e4b365b171d2">


### Slurm
We provide slurm files to submit jobs to a remote cluster in order to run the large scale experiments defined in this work. We provide both files to run array jobs of different parameter configurations, as well as a slurm job to create a Ray cluster, and perform hyperparameter optimization with HEBO or Optuna to find an optimal set of hyperparameters.

## Primal-Dual Approximate policy iteration

### Algorithm definition
![Reference Image](assets/img/pd-api.png)


### Policies and convergence curves
![Reference Image](assets/img/pd_api_stochastic_5x5.png)

![Reference Image](assets/img/pd_api_stochastic_8x8.png)


<!-- 
### XPPO
![Reference Image](assets/img/xppo.png)

### XSAC
![Reference Image](assets/img/exact_xsac.png)

### XTD3
![Reference Image](assets/img/exact_xtd3.png)
 -->
