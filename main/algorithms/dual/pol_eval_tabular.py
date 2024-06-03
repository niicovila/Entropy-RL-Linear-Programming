import multiprocessing
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from copy import deepcopy
import csv
import os
import random
import time
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import abc
from functools import cached_property


def plot_policy_with_heatmap(policy, value_function, env_image, M=5):
    # Compute U and V components for the quiver plot
    U = (policy[:, 2] - policy[:, 0]).reshape(M, M)
    V = (policy[:, 3] - policy[:, 1]).reshape(M, M)
    
    # Reshape the value function to match the grid
    heatmap_values = value_function.reshape(M, M)
    
    # Create the meshgrid for plotting
    x = np.arange(M) + 0.5
    X, Y = np.meshgrid(x, x[::-1])

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the heatmap
    cax = ax.imshow(heatmap_values, cmap='gray', extent=[0, M, 0, M])
    fig.colorbar(cax, ax=ax, orientation='vertical')

    # Overlay the environment image if provided
    if env_image is not None:
        ax.imshow(env_image, extent=[0, M, 0, M], alpha=0.5)

    # Plot the policy using quiver
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=2, color='red')

    # Set axis limits and aspect ratio
    ax.set(xlim=[0, M], ylim=[0, M], aspect='equal')

    # Show the plot
    plt.show()

def plot_policy(policy, save_dir, env_image, M):

    U = (policy[:, 2] - policy[:, 0]).reshape(M, M)
    V = (policy[:, 3] - policy[:, 1]).reshape(M, M)

    x = np.arange(M) + 0.5
    X, Y = np.meshgrid(x, x[::-1])

    #U = np.ones(25).reshape(5,5)
    #V = U

    fig, ax = plt.subplots()


    ax.imshow(env_image, extent=[0, M, 0, M])

    ax.quiver(X, Y, U,V, angles='xy', scale_units='xy', scale=2)
    ax.set(xlim=[0,M], ylim=[0,M], aspect="equal")
    plt.savefig(save_dir)
    plt.close()

    
def compute_P(env):
    S = env.observation_space.n
    A = env.action_space.n

    P = P = np.zeros((S, A, S))

    for (state, state_data) in env.P.items():
        for (action, next_data) in state_data.items():
            for (prob, next_state, reward, terminated) in next_data:
                P[state, action, next_state] = prob

    return P


def compute_R(env):
    S = env.observation_space.n
    A = env.action_space.n

    R = np.zeros((S, A, S))

    for (state, state_data) in env.P.items():
        for (action, next_data) in state_data.items():
            for (prob, next_state, reward, terminated) in next_data:
                R[state, action, next_state] = reward

    return R


class InvalidatePolicy:
    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value
        instance.__dict__.pop("policy", None)

class Algorithm(abc.ABC):

    def __init__(self, env, df=0.95, seed=None):
        self.env = env
        self.df = df
        
        # useful as a shortcut
        self.S = env.observation_space.n
        self.A = env.action_space.n

        # initialize the random number generator
        self.rng = np.random.default_rng(seed)


    def step(self, t):
        raise NotImplementedError
    
class LoggingMixin:
    "Utility class which evaluates the current policy at each step\
     and saves the values in `_rewards`."
    
    def __init__(self, env, *args, log_every=10, **kwds):
        super().__init__(env, *args, **kwds)

        self._log_every = log_every
        self._I = np.eye(self.S)
        self._rewards = []

    def evaluate_policy(self, policy):
        mean_return = []
        for i in range(10):
            next_s, _ = self.env.reset()
            done = False

            while not done:
                action = self.rng.choice(self.A, p=policy[next_s])
                next_s, r, terminations, truncations, infos = self.env.step(action)
                done = np.logical_or(terminations, truncations)
            mean_return.append(r)
        return np.mean(mean_return)

    def step(self, t, K, optimizer):
        super().step(t, K, optimizer)

        if t % self._log_every == 0:
            r = self.evaluate_policy(self.policy)
            print(f"Step {t}: {r}")
            self._rewards.append(r)

        if t == (self.total_timesteps):
            r = self.evaluate_policy(self.policy)
            self._rewards.append(r)
            print(f"Step {t}: {r}")
            return r


def softmax(matrix, axis=1):
    max_val = np.max(matrix, axis=axis, keepdims=True)
    shifted_matrix = matrix - max_val
    exp_matrix = np.exp(shifted_matrix)
    sum_exp = np.sum(exp_matrix, axis=axis, keepdims=True)
    softmax_matrix = exp_matrix / sum_exp

    return softmax_matrix

def one_hot_encode(tensor, num_classes=25):
    one_hot = torch.zeros(tensor.size(0),  num_classes)
    one_hot = one_hot.scatter_(1, tensor.long().unsqueeze(-1), 1)
    return one_hot

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.M = self.env.observation_space.n
        self.df = args.df
        self.critic = nn.Sequential(
            nn.Linear(self.M, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        x = one_hot_encode(x)
        q = self.critic(x)
        q = torch.clamp(q, 0, 1/(1 - self.df))
        return q
    
    def get_values(self, x, policy):
        q = self(x)
        probs = torch.Tensor(policy[x])
        v = torch.sum(probs * q, dim=1).squeeze(-1)
        return q, v

    
class TabularBase(Algorithm):
    policy_sum = InvalidatePolicy()

    def __init__(self, env, **kwds):
        lr_q = lr_z = temp = kwds.pop("lr", 0.01)
        self.lr_q = kwds.pop("lr_q", lr_q)
        self.lr_z = kwds.pop("lr_z", lr_z)
        self.temp = kwds.pop("temp", temp)
        self.reg_coef = kwds.pop("reg_coef", 0.1)
        self.tabular = kwds.pop("tabular", False)
        self.N = kwds.pop("N", 25)
        self.total_timesteps = kwds.pop("total_timesteps", 100000)
        self.sampling_strategy = kwds.pop("sampling_strategy", "reset")
        self.average_critic = kwds.pop("average_critic", False)
        
        super().__init__(env, **kwds)


        if self.tabular:
            self.q = np.full((self.S, self.A), 0)

        else:
            self.q = QNetwork(env, self)
        self.policy = np.ones((self.S, self.A)) / (self.A)
        self.policy_sum = self.policy
        self.z = np.full((self.N), 1)

    @cached_property
    def P(self):
        return compute_P(self.env)

    @cached_property
    def r(self):
        R = compute_R(self.env)
        return (self.P * R).sum(2)

    @property
    def nu0(self):
        return self.env.initial_state_distrib
    
    @cached_property
    def policy(self):
        return self.policy_sum / self.policy_sum.sum(1)[:, np.newaxis]
    
    def reset_z(self):
        self.z = np.full((self.N), 1)

SampleDT = [(x, int) for x in ["s0", "s", "a", "r", "next_s"]]


def collect_sample(env, s, a, agent):
    s0 = env.s
    
    env.unwrapped.s = s
    next_s, r, terminations, truncations, infos = env.step(a)
    next_done = np.logical_or(terminations, truncations)
    if next_done:
        if agent.sampling_strategy == "reset":
            next_s, _ = agent.env.reset()
        elif agent.sampling_strategy == "random":
            next_s = agent.rng.choice(agent.S)
        elif agent.sampling_strategy == "sample":
            next_s = agent.sample_state()
    
    return np.rec.array((s0, s, a, r, next_s), dtype=SampleDT), next_s


class GenerativePD(TabularBase):

    def sample_action(self, state):
        p = self.policy[state]
        return self.rng.choice(self.A, p = p)
    
    def sample_state(self):
        if self.tabular:
            state_dist = np.sum(self.policy * self.q, axis=1) + 1e-8
            state_dist /= state_dist.sum()
        else:
            state_dist = np.sum(self.policy * self.q(torch.arange(self.S)).detach().numpy(), axis=1)
            state_dist /= state_dist.sum()

        return self.rng.choice(self.S, p = state_dist)


    def grad_z(self, q, batch_transitions):
        if self.tabular:
            s = np.array([w.s for w in batch_transitions])
            a = np.array([w.a for w in batch_transitions])
            next_s = np.array([w.next_s for w in batch_transitions])
            r = np.array([w.r for w in batch_transitions])

            value_next_s = np.sum(self.policy * q, axis=1)
            v = value_next_s[next_s]
            grad = (r + self.df * v - q[s, a]) - self.reg_coef * np.log(self.z)

            return grad
        else:
            grad = torch.zeros(self.N)
            states = torch.tensor([int(w.s) for w in batch_transitions])
            next_states = torch.tensor([int(w.next_s) for w in batch_transitions])
            rewards = torch.tensor([float(w.r) for w in batch_transitions])
            actions = torch.tensor([int(w.a) for w in batch_transitions])
            z_values = torch.tensor([float(z) for z in self.z])

            q_values, v_values = self.q.get_values(states, self.policy)
            q_selected = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1).detach().numpy()
            v_next_values = self.q.get_values(next_states, self.policy)[1].detach().numpy()

            errors = (rewards + self.df * v_next_values - q_selected) - self.reg_coef * np.log(z_values)
            grad += errors

            return grad.numpy()
    
    def grad_q(self, z, batch_transitions):
        s0 = np.array([w.s0 for w in batch_transitions])
        s = np.array([w.s for w in batch_transitions])
        a = np.array([w.a for w in batch_transitions])
        next_s = np.array([w.next_s for w in batch_transitions])
        z = np.array([self.z[i] for i, _ in enumerate(batch_transitions)])

        grad = np.zeros((self.S, self.A))

        first_action = np.array([self.policy[s0_i].argmax() for s0_i in s0])
        next_a = np.array([self.policy[next_s_i].argmax() for next_s_i in next_s])

        np.add.at(grad, (s0, first_action), (1 - self.df))
        np.add.at(grad, (s, a), -z)
        np.add.at(grad, (next_s, next_a), self.df * z)

        return grad
    
    def step_q(self, batch_transitions, optimizer):
        if self.tabular:
            grad = self.grad_q(self.z, batch_transitions)
            q = self.q - self.lr_q * grad / self.N
            q_min = 0
            q_max = 1/(1-self.df)
            self.q = np.clip(q, q_min, q_max) 
        else:
            states = torch.tensor([int(w.s) for w in batch_transitions])
            next_states = torch.tensor([int(w.next_s) for w in batch_transitions])
            rewards = torch.tensor([float(w.r) for w in batch_transitions])
            actions = torch.tensor([int(w.a) for w in batch_transitions])
            z_values = torch.tensor([float(z) for z in self.z])

            q_values, v_values = self.q.get_values(states, self.policy)
            q_selected = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            v_next_values = self.q.get_values(next_states, self.policy)[1]
            target_values = rewards + self.df * v_next_values

            error = torch.mean(z_values * (target_values - q_selected)  + (1 - self.df) * v_values) #- self.reg_coef * (z_values * torch.log(z_values) - z_values)

            optimizer.zero_grad()
            error.backward()
            optimizer.step()
    
    def step_z(self, batch_transitions):
        grad = self.grad_z(self.q, batch_transitions)
        z = self.z * np.exp(self.lr_z * grad / self.N)
        return z
    
    def step_policy(self, q):
        if self.tabular:
            q_values = self.q
            update = softmax(q_values * self.temp)
            policy_probs =  self.policy * update
            policy = policy_probs / policy_probs.sum(1)[:, np.newaxis]
            return policy
        else:
            q_values = self.q(torch.arange(self.S)).detach()
            update = F.softmax(q_values * self.temp, dim=1).numpy()
            policy_probs = self.policy * update
            policy = policy_probs / policy_probs.sum(1)[:, np.newaxis]
            return policy
    
    def step(self, t, K, optimizer):  
        batch_transitions = []

        if self.sampling_strategy == "reset":
            next_obs, _ = self.env.reset()
        elif self.sampling_strategy == "random":
            next_obs = self.rng.choice(self.S)
        elif self.sampling_strategy == "sample":
            next_obs = self.sample_state()

        for t in range(self.N):
            a = self.sample_action(next_obs)
            self.sample, next_obs = collect_sample(self.env, next_obs, a, self)
            batch_transitions.append(self.sample)

        weights_after_each_epoch = []
        q_sum = np.zeros((self.S, self.A))

        for k in range(K):
            self.step_q(batch_transitions, optimizer)
            self.z = self.step_z(batch_transitions)
            if self.tabular:
                q_sum += self.q
            else:
                weights_after_each_epoch.append({key: val.cpu().numpy() for key, val in self.q.state_dict().items()})

        if self.average_critic:
            if self.tabular:
                self.q = q_sum / K
            else:
                avg_weights = {}
                for key in weights_after_each_epoch[0].keys():
                    avg_weights[key] = np.mean([T[key] for T in weights_after_each_epoch], axis=0)
                self.q.load_state_dict({key: torch.tensor(val) for key, val in avg_weights.items()})

        self.policy = self.step_policy(self.q)
        self.policy_sum = self.policy
        self.reset_z()


class LoggingGenerativePD(LoggingMixin, GenerativePD):
    pass

def tune_pol_eval(config):
    ## Run the agent
    total_timesteps = 5000
    K = config["K"]
    lr_q = config["lr_q"]
    lr_z = config["lr_z"]
    temp = config["temp"]
    reg_coef = config["reg_coef"]
    N = config["batch_size"]
    tabular = config["tabular"]
    total_timesteps = total_timesteps

    RANDOM_SEED = config['seed']
    seed_seq = np.random.SeedSequence(RANDOM_SEED)
    MAP_SEED, ENV_SEED, ALG_SEED = seed_seq.spawn(3)
    MAP_SEED = int(MAP_SEED.generate_state(1)[0])
    ENV_SEED = int(ENV_SEED.generate_state(1)[0])
    M = MAP_SIZE = 5
    FROZEN_PROBABILITY=0.9

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    map = generate_random_map(
        size=MAP_SIZE,
        p=FROZEN_PROBABILITY,
        seed=ENV_SEED,
    )

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="rgb_array",
        desc=map,
    )

    env.reset(seed=ENV_SEED) # set the random seed
    env_image = env.render()
    agent = LoggingGenerativePD(
        env,
        lr_q = lr_q,
        lr_z = lr_z,
        log_every = 1000,
        reg_coef = reg_coef,
        temp = temp,
        tabular = tabular,
        sampling_strategy = config["sampling_strategy"],
        average_critic = config["average_critic"],
        N = N,
        total_timesteps = total_timesteps,
        seed = ALG_SEED
    )

    if agent.tabular:
        optimizer = None

    else:
        optimizer = getattr(optim, config["optimizer"])(
                list(agent.q.parameters()), lr=lr_q
            )
    print("Total timesteps: ", total_timesteps)

    for t in range(1, total_timesteps + 1):
        if config["anneal_lr"]:
            frac = 1.0 - (t - 1.0) / total_timesteps
            if tabular:
                agent.lr_q = frac * config["lr_q"]
            else:
                lrnow = frac * config["lr_q"]
                optimizer.param_groups[0]["lr_q"] = lrnow

        agent.step(t, K, optimizer)
    
    return agent._rewards

config = {
    "K": [10, 20, 25, 30, 40, 50, 65],
    "lr_q" : list(np.linspace(0.0001, 0.5, 1000)),
    "lr_z" : list(np.linspace(0.0001, 0.5, 1000)),
    "temp" : list(np.linspace(0.01, 0.75, 100)),
    "optimizer" : ["SGD", "Adam"],
    "reg_coef" : list(np.linspace(0.01, 2, 1000)),
    "batch_size": [12, 24, 48, 96],
    "tabular": [True, False],
    "anneal_lr": [True, False],
    "average_critic": [True, False],
    "sampling_strategy": ["sample", "random", "reset"],
}

def create_samples(n=10):
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
    csv_file = "samples_pol_eval_final.csv"

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)

    print(f"Samples written to {csv_file}")  

def run_config(args):
    row, seed = args
    config = {}
    for column in row.keys():
        field_value = row[column] 

        if isinstance(field_value, np.bool_):
            field_value = bool(field_value)

        config[column] = field_value

    config['seed'] = seed   

    return tune_pol_eval(config), row['index']

def save_results(results, prefix):
    rewards, trial_ids = zip(*results)

    rewards_df = []
    for r in rewards:
        rewards_df.append(pd.DataFrame(r, columns=['Reward']))
        rewards_df[-1]['Step'] = np.arange(0, len(r) * 1000, 1000)
    
    combined_df = pd.concat(rewards_df)
    average_reward = combined_df.groupby('Step')['Reward'].mean()    

    save_dir = f'./data_{prefix}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    average_reward.to_csv(f'{save_dir}/dual_{trial_ids[0]}.csv')

    

if __name__ == "__main__":
    current_dir = os.getcwd()
    num_seeds = 3
    df = pd.read_csv(current_dir + '/samples_pol_eval_final.csv')

    for i in range(len(df)):
        row_index = int(i)
        row= df.iloc[row_index]

        tasks = [(row, seed+1) for seed in range(num_seeds)]
        start = time.time()

        with multiprocessing.Pool(processes=num_seeds) as pool:
            results = pool.map(run_config, tasks)

        print(results)
        print("End time: ", time.time() - start)

        save_results(results, row['index'])