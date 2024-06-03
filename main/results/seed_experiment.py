import os
import sys
import pandas as pd
from algorithms import tune_elbe
from algorithms import tune_saddle
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing

# from ray.util.multiprocessing import Pool

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def run_config(args):
    row, df, seed = args
    config = {}
    for column in df.columns:
        if column != 'config/__trial_index__':
            if column.startswith('config/'):
                field_name = column.split('/', 1)[1] 
                field_value = row[column] 

                if isinstance(field_value, np.bool_):
                    field_value = bool(field_value)

                config[field_name] = field_value

    config['seed'] = seed   
    config['save_learning_curve'] = True
    config['eta'] = None

    try:
        return tune_saddle(config), row['trial_id']
    except: 
        return pd.DataFrame({'Step': range(100000), 'Reward': [-2000] * 100000}), row['trial_id']

def create_plots(results, procedure, folder_name):
    rewards, trial_ids = zip(*results)

    unique_steps = set()
    for df in rewards:
        unique_steps.update(df['Step'].unique())

    dfs = []
    # Step 2: Interpolate rewards for missing steps in each dataframe
    for i, df in enumerate(rewards):
        df.set_index('Step', inplace=True)  # Set 'Step' as index for easier interpolation
        df = df.reindex(unique_steps)  # Reindex to include all unique steps
        df['Reward'] = df['Reward'].interpolate()  # Interpolate rewards for missing steps
        df.reset_index(inplace=True)  # Reset index after interpolation
        dfs.append(df)
    
    combined_df = pd.concat(dfs)

    # Calculate the average reward grouped by 'Step'
    average_reward = combined_df.groupby('Step')['Reward'].mean()    
    rolling_average = average_reward.rolling(window=7).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(average_reward.index, average_reward, label='Original Reward', color='gray', alpha=0.7)
    plt.plot(rolling_average.index, rolling_average, label='Rolling Average (Window=7)', color='blue')

    plt.title('Average Episodic Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.savefig(f'./plots/{procedure}_{folder_name}_{trial_ids[0]}.png')
    average_reward.to_csv(f'./data/{procedure}_{folder_name}_{trial_ids[0]}.csv')


if __name__ == '__main__':
    current_dir = os.getcwd()
    folder = 'folder_path'
    df = pd.read_csv(current_dir + folder)   
    num_seeds = 3

    row_index = 0
    row= df.iloc[row_index]
    tasks = [(row, df, seed+1) for seed in range(num_seeds)]
    start = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(run_config, tasks)
    pool.terminate()

    create_plots(results, procedure='elbe', folder_name='cartpole')

    print("End time: ", time.time() - start)