import os
import sys
import random
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing
from multiprocessing import Pool
from algorithms import tune_elbe

config_elbe = {
    "exp_name": "QREPS",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,

    "env_id": "LunarLander-v2",

    # Algorithm
    "total_timesteps": 100000,
    "num_envs": 32,
    "gamma": 0.99,

    "total_iterations": [512, 1024, 2048, 4096],
    "num_minibatches": [4, 8, 16, 32, 64],
    "update_epochs": [10, 25, 50, 100, 150, 300],

    "alpha": [2, 4, 8, 12, 32, 64, 100],  
    "eta": None,  

    # Learning rates
    "policy_lr": [3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003],
    "q_lr": [3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003],
    "anneal_lr": [True, False],

    # Layer Init
    "layer_init":  "kaiming_uniform",

    # Architecture
    "policy_activation": "Tanh",
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "actor_last_layer_std": 0.01,

    "q_activation": "Tanh",
    "q_num_hidden_layers": 4,
    "q_hidden_size": 128,
    "q_last_layer_std": 1.0,
    "use_policy": True,

    # Optimization
    "q_optimizer": ["Adam", "SGD", "RMSprop"],
    "actor_optimizer": ["Adam", "SGD", "RMSprop"],
    "eps": 1e-8,

    # Options
    "average_critics": [True, False],
    "normalize_delta": False,
    "use_kl_loss": True,
    "q_histogram": False,
    "gae": False,
    "gae_lambda": 0.95,

    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0, 

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}

def create_samples(n=10):
    samples = []
    for i in range(n):
        sample = {"index": i + 1} 
        for key, value in config_elbe.items():
            if isinstance(value, list):
                sample[key] = random.choice(value)
            else:
                sample[key] = value
        samples.append(sample)

    # Writing to CSV
    csv_columns = list(samples[0].keys())
    csv_file = "samples.csv"

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)

    print(f"Samples written to {csv_file}")  

def save_results(results, prefix):
    rewards, trial_ids = zip(*results)

    unique_steps = set()
    for df in rewards:
        unique_steps.update(df['Step'].unique())

    dfs = []
    for i, df in enumerate(rewards):
        df.set_index('Step', inplace=True)
        df = df.reindex(unique_steps)
        df['Reward'] = df['Reward'].interpolate()
        df.reset_index(inplace=True)
        dfs.append(df)
    
    combined_df = pd.concat(dfs)
    average_reward = combined_df.groupby('Step')['Reward'].mean()    

    save_dir = f'./ray_tune/data_{prefix}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    average_reward.to_csv(f'{save_dir}/randsearch_{prefix}_{trial_ids[0]}.csv')


def run_config(args):
    row, df, seed = args
    config = {}
    for column in df.columns:
        field_value = row[column] 

        if isinstance(field_value, np.bool_):
            field_value = bool(field_value)

        config[column] = field_value

    config['seed'] = seed   
    config['save_learning_curve'] = True
    # print(config["eta"])
    # if not isinstance(config["eta"], float):
    #     print(config["eta"])
    config['eta'] = None

    return tune_elbe(config), row['index']

if __name__ == '__main__':
    
    print(os.cpu_count())
    row_index = int(sys.argv[1])
    prefix = (sys.argv[2])

    current_dir = os.getcwd()
    num_seeds = 3
    df = pd.read_csv(current_dir + f'/ray_tune/{prefix}.csv')
    
    row= df.iloc[row_index]

    tasks = [(row, df, seed+1) for seed in range(num_seeds)]
    start = time.time()

    with multiprocessing.Pool(processes=num_seeds) as pool:
        results = pool.map(run_config, tasks)

    print(results)
    pool.terminate()
    print("End time: ", time.time() - start)

    save_results(results, prefix)