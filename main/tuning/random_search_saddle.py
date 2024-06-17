import os
import sys
import random
import csv
import numpy as np
import pandas as pd
import time
import multiprocessing
from algorithms import tune_saddle

def create_samples(config, n=10):
    samples = []
    for i in range(n):
        sample = {"index": i + 1} 
        for key, value in config.items():
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
    config['eta'] = None

    return tune_saddle(config), row['index']

if __name__ == '__main__':
    
    print(os.cpu_count())
    row_index = int(sys.argv[1])
    prefix = (sys.argv[2])

    current_dir = os.getcwd()
    num_seeds = 3
    df = pd.read_csv(current_dir + f'{prefix}.csv')
    
    row= df.iloc[row_index]

    tasks = [(row, df, seed+1) for seed in range(num_seeds)]
    start = time.time()

    with multiprocessing.Pool(processes=num_seeds) as pool:
        results = pool.map(run_config, tasks)

    print(results)
    pool.terminate()
    print("End time: ", time.time() - start)

    save_results(results, prefix)