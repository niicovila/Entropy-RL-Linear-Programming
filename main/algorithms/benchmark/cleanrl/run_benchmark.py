from matplotlib import pyplot as plt
import pandas as pd
import glob
import subprocess
import asyncio

envs = ["Acrobot-v1"]
seeds = [1, 2, 3]

commands_ppo = [f"poetry run python cleanrl/ppo.py --total_timesteps 100000 --env_id {env} --seed {seed}" for env in envs for seed in seeds]
commands_sac = [f"poetry run python cleanrl/sac_discrete.py --total_timesteps 100000 --env_id {env} --seed {seed}" for env in envs for seed in seeds]
commands_dqn = [f"poetry run python cleanrl/dqn.py --total_timesteps 100000 --env_id {env} --seed {seed}" for env in envs for seed in seeds]
commands_c51 = [f"poetry run python cleanrl/c51.py --total_timesteps 100000 --env_id {env} --seed {seed}" for env in envs for seed in seeds]

async def run_command(command):
    process = await asyncio.create_subprocess_shell(command)
    await process.wait()

async def run_commands(commands):
    tasks = [run_command(command) for command in commands]
    await asyncio.gather(*tasks)

# Run the commands asynchronously
asyncio.run(run_commands(commands_ppo))
asyncio.run(run_commands(commands_sac))
asyncio.run(run_commands(commands_dqn))
asyncio.run(run_commands(commands_c51))


for env in envs:
    dfs_ppo_lun = []
    for file in glob.glob('outputs_ppo/*.csv'):
        if env in file:
            df = pd.read_csv(file)
            dfs_ppo_lun.append(df)


    dfs_sac_acro = []
    for file in glob.glob('outputs_sac/*.csv'):
        if env in file:
            df = pd.read_csv(file)
            dfs_sac_acro.append(df)



    dfs_dqn_lun = []
    for file in glob.glob('outputs_dqn/*.csv'):
        if env in file:
            df = pd.read_csv(file)
            dfs_dqn_lun.append(df)


    dfs_c51_lun = []
    for file in glob.glob('outputs_c51/*.csv'):
        if env in file:
            df = pd.read_csv(file)
            dfs_c51_lun.append(df)


    def create_plots(rewards, algo, env_id):
        
        unique_steps = set()
        for df in rewards:
            unique_steps.update(df['Step'].unique())
        print(rewards)
        combined_df = pd.concat(rewards)
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

        plt.savefig(f'plot_{algo}_{env_id}.png')
        average_reward.to_csv(f'average_reward_{algo}_{env_id}.csv')

    create_plots(dfs_ppo_lun, 'ppo', env)

    create_plots(dfs_sac_acro, 'sac', env)

    create_plots(dfs_dqn_lun, 'dqn', env)

    create_plots(dfs_c51_lun, 'c51', env)

