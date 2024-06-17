import sys
from matplotlib.ticker import ScalarFormatter
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import csv
import os
import random
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import abc
from functools import cached_property

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
    
class LoggingMixinCanonicalPD:
    "Utility class which evaluates the current policy at each step\
     and saves the values in `_rewards`."
    
    def __init__(self, env, *args, log_every=10, **kwds):
        super().__init__(env, *args, **kwds)

        self._log_every = log_every
        self._I = np.eye(self.S)
        self._rewards = []


    def evaluate_policy(self, policy):
        P = (self.P * policy.reshape(self.S, self.A, 1)).sum(1)
        r = (self.r * policy).sum(1)

        v = np.linalg.inv(self._I - self.df * P) @ r
        return (1-self.df) * self.nu0 @ v

    def step(self, t):
        super().step(t)

        if t % self._log_every == 0:
            r = self.evaluate_policy(self.policy)
            self._rewards.append(r)

class LoggingMixinDualQ:
    "Utility class which evaluates the current policy at each step\
     and saves the values in `_rewards`."
    
    def __init__(self, env, *args, log_every=10, **kwds):
        super().__init__(env, *args, **kwds)

        self._log_every = log_every
        self._I = np.eye(self.S)
        self._rewards = []


    def evaluate_policy(self, policy):
        P = (self.P * policy.reshape(self.S, self.A, 1)).sum(1)
        r = (self.r * policy).sum(1)

        v = np.linalg.inv(self._I - self.df * P) @ r
        return (1-self.df) * self.nu0 @ v

    def step(self, t):
        super().step(t)

        if t % self._log_every == 0:
            r = self.evaluate_policy(self.policy)
            self._rewards.append(r)

SampleDT = [(x, int) for x in ["s0", "s", "a", "r", "next_s"]]


def collect_sample(env, s, a):
    s0 = env.s
    
    env.unwrapped.s = s # the hours I spent before I realized I had to put `unwrapped` here. You have no idea.
    next_s, r, *_ = env.step(a)
    return np.rec.array((s0, s, a, r, next_s), dtype=SampleDT)


class CanonicalLP(Algorithm):

    # Makes sure the policy is deleted every time mu_sum is assigned
    mu_sum = InvalidatePolicy()

    def __init__(self, env, **kwds):
        lr_v = lr_mu = kwds.pop("lr", 0.01)
        self.lr_v = kwds.pop("lr_v", lr_v)
        self.lr_mu = kwds.pop("lr_mu", lr_mu)
        
        super().__init__(env, **kwds)

        # initialize variables
        self.v = np.zeros(self.S)
        self.mu = np.ones((self.S, self.A)) / (self.S * self.A)
        self.mu_sum = self.mu


    # @cached_property computes the attribute on first access and then caches it
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
        return self.mu_sum / self.mu_sum.sum(1).reshape(-1, 1)

class CanonicalLPAlgorithm(CanonicalLP):

    def sample_action(self, state):
        p = self.mu[state]
        return self.rng.choice(self.A, p = p / p.sum())

    def sample_state(self):
        return self.rng.choice(self.S, p = self.mu.sum(1))
    
    def step_v(self):
        grad = self.grad_v(self.mu)

        v = self.v - self.lr_v * grad

        v_min = 0
        v_max = 1/(1-self.df)
        return np.clip(v, v_min, v_max)


    def step_mu(self):
        grad = self.grad_mu(self.v)

        p = self.mu * np.exp(self.lr_mu * grad)
        mu = p / p.sum()
        return mu

    def grad_mu(self, v):
        w = self.sample

        grad = np.zeros((self.S, self.A))
        grad[w.s, w.a] = (w.r + self.df * v[w.next_s] - v[w.s]) * self.S * self.A #/ self.mu[w.s, w.a] #
        return grad

    def grad_v(self, mu):
        w = self.sample

        grad = np.zeros(self.S)
        grad[w.s0] = (1 - self.df)
        grad[w.next_s] += self.df
        grad[w.s] -= 1
        #grad -= mu.sum(1)
        #grad[w.next_s] += self.df * mu[w.s, w.a] - mu[w.next_s].sum()
        return grad


    def step(self, t):  
        # update for mu
        s = self.rng.choice(self.S)
        a = self.rng.choice(self.A) 
        self.sample = collect_sample(self.env, s, a)
        
        mu = self.step_mu()
        
        # reset the env to get a fresh sample from the initial distrib
        self.env.reset()
        
        # update for v
        s = self.sample_state()
        a = self.sample_action(s)
        self.sample = collect_sample(self.env, s, a)

        v = self.step_v()

        
        # set the variables
        self.v = v
        self.mu = mu
        self.mu_sum += mu

class LoggingCanonicalPD(LoggingMixinCanonicalPD, CanonicalLPAlgorithm):
    pass

class DualQ(Algorithm):

    # Makes sure the policy is deleted every time mu_sum is assigned
    mu_sum = InvalidatePolicy()

    def __init__(self, env, **kwds):
        lr_q = lr_mu = kwds.pop("lr", 0.01)
        self.lr_q = kwds.pop("lr_q", lr_q)
        self.lr_mu = kwds.pop("lr_mu", lr_mu)
        
        super().__init__(env, **kwds)

        # initialize variables
        self.q = np.full((self.S, self.A), 0)
        self.mu = np.ones((self.S, self.A)) / (self.S * self.A)
        self.mu_sum = self.mu


    # @cached_property computes the attribute on first access and then caches it
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
        return self.mu_sum / self.mu_sum.sum(1).reshape(-1, 1)

class DualQAlgorithm(DualQ):

    def sample_action(self, state):
        p = self.mu[state]
        return self.rng.choice(self.A, p = p / p.sum())

    def sample_state(self):
        return self.rng.choice(self.S, p = self.mu.sum(1))
    
    def grad_mu(self, q):
        w = self.sample
        grad = np.zeros((self.S, self.A))
        v = np.sum(self.policy[w.next_s] * q[w.next_s])
        grad[w.s, w.a] = (w.r + self.df * v - q[w.s, w.a]) * self.S * self.A 
        return grad

    def grad_q(self, mu):
        w = self.sample
        grad = np.zeros((self.S, self.A))
        # first_action = self.policy[w.s0].argmax()
        first_action = self.rng.choice(self.A, p = self.policy[w.s0])
        # next_a = self.policy[w.next_s].argmax()
        next_a = self.rng.choice(self.A, p = self.policy[w.next_s])

        grad[w.s0, first_action] = (1 - self.df)
        grad[w.s, w.a] -= 1 
        grad[w.next_s, next_a] += self.df 

        return grad
    
    def step_q(self):
        grad = self.grad_q(self.mu)
        q = self.q - self.lr_q * grad

        q_min = 0
        q_max = 1/(1-self.df)
        return np.clip(q, q_min, q_max)
    
    def step_mu(self):
        grad = self.grad_mu(self.q)

        p = self.mu * np.exp(self.lr_mu * grad)
        mu = p / p.sum()
        return mu

    def step(self, t):  
        s = self.rng.choice(self.S)
        a = self.rng.choice(self.A) 
        self.sample = collect_sample(self.env, s, a)

        mu = self.step_mu()

        self.env.reset()
        s = self.sample_state()
        a = self.sample_action(s)
        self.sample = collect_sample(self.env, s, a)

        q = self.step_q()

        self.q = q
        self.mu = mu
        self.mu_sum += mu

class LoggingDualQ(LoggingMixinDualQ, DualQAlgorithm):
    pass

class LoggingMixinDualSampler:
    "Utility class which evaluates the current policy at each step\
     and saves the values in `_rewards`."
    
    def __init__(self, env, *args, log_every=10, **kwds):
        super().__init__(env, *args, **kwds)

        self._log_every = log_every
        self._I = np.eye(self.S)
        self._rewards = []

    def evaluate_policy(self, policy):
        P = (self.P * policy.reshape(self.S, self.A, 1)).sum(1)
        r = (self.r * policy).sum(1)

        v = np.linalg.inv(self._I - self.df * P) @ r
        return (1-self.df) * self.nu0 @ v

    def step(self, t, K, optimizer):
        super().step(t, K, optimizer)
        if t % self._log_every == 0:
            r = self.evaluate_policy(self.policy)
            print(f"Step {t}: {r}")
            self._rewards.append(r)

class SamplerDual(Algorithm):
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
        self.average_critic = kwds.pop("average_critic", False)
        self.v_mode = kwds.pop("V", "dual")
        self.update = kwds.pop("update", "batch")
        self.q_init = kwds.pop("q_init", 'max')
        self.sampling = kwds.pop("sampling", 'random')
        
        super().__init__(env, **kwds)

        if self.tabular:
            if self.q_init == 'max':
                self.q = np.full((self.S, self.A), 1 / (1-self.df))
            elif self.q_init == 'dist':
                self.q = np.full((self.S, self.A), 1 / (self.S*self.A))
            elif self.q_init == 'zero':
                self.q = np.zeros((self.S, self.A))

        else:
            self.q = QNetwork(env, self)
        self.policy = np.ones((self.S, self.A)) / (self.A)
        self.policy_sum = self.policy

        if self.v_mode == 'dual': self.z = np.full((self.N), 1)
        else: self.z = np.full((self.N), 1/self.N)

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
        if self.v_mode == 'dual': self.z = np.full((self.N), 1)
        else: self.z = np.full((self.N), 1/self.N)

SampleDTDual = [(x, int) for x in ["s0", "s", "a", "r", "next_s", "done"]]

def collect_sample_batch(env, s0, s, a, agent):    
    env.unwrapped.s = s
    next_s, r, terminations, truncations, infos = env.step(a)
    next_done = np.logical_or(terminations, truncations)
    sample = np.rec.array((s0, s, a, r, next_s, next_done), dtype=SampleDTDual)
    
    if agent.sampling == 'state_dist':
        s0 = agent.sample_state()
        next_s = s0

    if agent.sampling == 'reset':
        if next_done: 
            s0, _ = env.reset()
            next_s = s0
    else: 
        s0 = agent.rng.choice(agent.S)
        next_s = s0

    return sample, s0, next_s

class SamplerDualAlgorithm(SamplerDual):

    def sample_action(self, state):
        p = self.policy[state]
        return self.rng.choice(self.A, p = p/p.sum())

    def sample_state(self):
        v = np.sum(self.policy * self.q, axis=1)   
        if v.sum() > 0: 
            return self.rng.choice(self.S, p = v/v.sum())
        else:
            return self.rng.choice(self.S)
    
    def grad_z(self, q, batch_transitions):
        if self.tabular:
            s = np.array([w.s for w in batch_transitions])
            a = np.array([w.a for w in batch_transitions])
            next_s = np.array([w.next_s for w in batch_transitions])
            r = np.array([w.r for w in batch_transitions])
            next_done = np.array([w.done for w in batch_transitions])
            
            if self.v_mode == 'dual':
                value_next_s = np.sum(self.policy * q, axis=1)
                v = value_next_s[next_s]
            elif self.v_mode == 'qreps':
                q_max = np.max(q / self.reg_coef, axis=1, keepdims=True)
                v = self.reg_coef * (
                    np.log(np.sum(self.policy * np.exp((q / self.reg_coef) - q_max), axis=1)) + q_max.flatten()
                )
                v = v[next_s]
            v_min = 0
            if self.df != 1: v_max = 1/(1-self.df)
            else: v_max = 1.0
            v = np.clip(v, v_min, v_max) 
            grad = ((r + self.df * v * (1 - next_done) - q[s, a]) - self.reg_coef * np.log(self.z+1e-8))
            return grad * self.N
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
        grad = np.zeros((self.S, self.A))

        if self.update == 'batch':
            s0 = np.array([w.s0 for w in batch_transitions])
            s = np.array([w.s for w in batch_transitions])
            a = np.array([w.a for w in batch_transitions])
            next_s = np.array([w.next_s for w in batch_transitions])
            z = np.array([self.z[i] for i, _ in enumerate(batch_transitions)])
            next_dones = np.array([w.done for w in batch_transitions])

            first_action = np.array([self.policy[s0_i].argmax() for s0_i in s0])
            next_a = np.array([self.policy[next_s_i].argmax() for next_s_i in next_s])

            np.add.at(grad, (s0, first_action), (1 - self.df))
            np.add.at(grad, (s, a), -z)
            np.add.at(grad, (next_s, next_a), self.df * z * (1-next_dones))

        elif self.update == 'stochastic':
            index = self.rng.choice(self.N, p = self.z/self.z.sum())
            s0 = batch_transitions[index].s0
            s = batch_transitions[index].s
            a = batch_transitions[index].a
            next_s = batch_transitions[index].next_s
            next_a = self.sample_action(next_s)
            first_action = self.sample_action(s0)
            next_dones = batch_transitions[index].done

            grad[s0, first_action] += 1 - self.df
            grad[s, a] -= 1
            grad[next_s, next_a] += self.df * (1 - next_dones)

        return grad #/ self.N
    
    def step_q(self, batch_transitions, optimizer):
        if self.tabular:
            grad = self.grad_q(self.z, batch_transitions)
            if self.update == 'batch':
                q = self.q - self.lr_q * grad * self.z.sum()
            elif self.update == 'stochastic':
                q = self.q - self.lr_q * grad
            q_min = 0
            if self.df != 1: q_max = 1/(1-self.df)
            else: q_max = 1.0
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
        z = self.z * np.exp(self.lr_z * grad)
        if self.v_mode == 'qreps': 
            z = z / z.sum()
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
        self.env.reset()
        if self.sampling == 'state_dist':
            next_obs = self.sample_state()
        elif self.sampling == 'random': next_obs = self.rng.choice(self.S)
        elif self.sampling == 'reset': next_obs, _ = self.env.reset()
        first_obs = next_obs
        for t in range(self.N):
            a = self.sample_action(next_obs)
            self.sample, first_obs, next_obs = collect_sample_batch(self.env, first_obs, next_obs, a, self)
            batch_transitions.append(self.sample)

        weights_after_each_epoch = []
        q_sum = np.zeros((self.S, self.A))

        for k in range(K):
            self.z = self.step_z(batch_transitions)
            self.step_q(batch_transitions, optimizer)

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

class LoggingSamplerDual(LoggingMixinDualSampler, SamplerDualAlgorithm):
    pass

def tune_pol_eval(config):
    ## Run the agent
    print(config)
    total_timesteps = 200000 // config["batch_size"]
    K = config["K"]
    lr_q = config["lr_q"]
    lr_z = config["lr_z"]
    temp = config["temp"]
    reg_coef = config["reg_coef"]
    N = config["batch_size"]
    tabular = config["tabular"]
    df = config["df"]
    total_timesteps = total_timesteps

    RANDOM_SEED = config['seed']
    seed_seq = np.random.SeedSequence(RANDOM_SEED)
    MAP_SEED, ENV_SEED, ALG_SEED = seed_seq.spawn(3)
    MAP_SEED = int(MAP_SEED.generate_state(1)[0])
    ENV_SEED = int(ENV_SEED.generate_state(1)[0])
    M = MAP_SIZE = 5
    FROZEN_PROBABILITY=0.8
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    map = generate_random_map(
        size=MAP_SIZE,
        p=FROZEN_PROBABILITY,
        seed=ENV_SEED,
    )
    _map = [
        "FFFHF",
        "FHFFH",
        "FFHFF",
        "HFHFF",
        "FFSHG"
    ]

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="rgb_array",
        desc=map,
    )

    env.reset(seed=ENV_SEED) # set the random seed
    env_image = env.render()

    agent = LoggingSamplerDual(
        env,
        lr_q = lr_q,
        lr_z = lr_z,
        log_every = 100,
        reg_coef = reg_coef,
        temp = temp,
        tabular = tabular,
        average_critic = config["average_critic"],
        N = N,
        df = df,
        V = config['V'],
        q_init = config['q_init'],
        update = config['update'],
        sampling = config['sampling'],
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
                agent.lr_z = frac * config["lr_z"]
            else:
                lrnow = frac * config["lr_q"]
                optimizer.param_groups[0]["lr_q"] = lrnow
        agent.step(t, K, optimizer)

        # if t%5000 == 0:
        #     v = np.sum(agent.policy * agent.q, axis=1)
        #     plot_policy_with_heatmap(agent.policy, v, env_image, M)

    return agent

config = {
    "K": [3, 5, 8, 10, 15, 20, 25, 30],
    "lr_q" : [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.05],
    "lr_z" : [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.05],
    "temp" : [0.003, 0.005, 0.01, 0.03, 0.04, 0.05, 0.1, 0.2, 0.4],
    "optimizer" : "Adam",
    "reg_coef" : [0.1, 0.5, 1, 2, 2.5, 4, 8, 10],
    "batch_size": [10, 15, 25, 40, 50],
    "tabular": True,
    "anneal_lr": False,
    "average_critic": [True, False],
    "df" : 0.95,
    "V": "dual",
    "update": ["batch", "stochastic"],
    "q_init": ["max", "dist", "zero"],
    "sampling": ["state_dist", "random", "reset"],
}

def create_samples(n=10, seed=1):
    random.seed(seed)
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
    csv_file = "samples_pol_eval_v2.csv"

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

    save_dir = f'./data_dual'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    average_reward.to_csv(f'{save_dir}/dual_{trial_ids[0]}.csv')


if __name__ == "__main__":
    current_dir = os.getcwd()
    num_seeds = 10
    df_qreps = pd.read_csv(current_dir + '/samples_pol_eval_stochastic_qreps.csv')
    df_dual = pd.read_csv(current_dir + '/samples_pol_eval_stochastic.csv')

    best_configs_qreps = 305 #929
    # best_configs_dual = [842, 807, 878, 584, 579, 422, 424, 499, 758, 706, 592, 665]
    best_configs_dual = [842]
    # good (8x8): 486, 842, 878, 424, 706, 592
    # first good then bad: 807
    for config_dual in best_configs_dual:
        config_qreps = df_qreps.iloc[best_configs_qreps-1]
        config_dual = df_dual.iloc[config_dual-1]
        for seed in range(num_seeds):
            RANDOM_SEED = 670 + seed
            seed_seq = np.random.SeedSequence(RANDOM_SEED)
            MAP_SEED, ENV_SEED, ALG_SEED = seed_seq.spawn(3)
            MAP_SEED = int(MAP_SEED.generate_state(1)[0])
            ENV_SEED = int(ENV_SEED.generate_state(1)[0])
            M = MAP_SIZE = 5
            FROZEN_PROBABILITY=0.8
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)

            map = generate_random_map(
                size=MAP_SIZE,
                p=FROZEN_PROBABILITY,
                seed=ENV_SEED,
            )

            _map = [
                "FFFHF",
                "FHFFH",
                "FFHFF",
                "HFHFF",
                "FFSHG"
            ]

            env = gym.make(
                "FrozenLake-v1",
                is_slippery=False,
                render_mode="rgb_array",
                desc=map,
            )

            env.reset(seed=ENV_SEED) # set the random seed
            env_image = env.render()

            config_qreps['seed'] = RANDOM_SEED
            config_dual['seed'] = RANDOM_SEED

            print(config_dual)
            agent_dual = tune_pol_eval(config_dual)
            agent_qreps = tune_pol_eval(config_qreps)

            lr = 0.0031
            n = 200000
            agent_canonical = LoggingCanonicalPD(
                env,
                lr_v = lr,
                lr_mu = lr / 3,
                log_every = 100,
                seed = ALG_SEED
            )
            for t in range(1, n):
                agent_canonical.step(t)

            # lr_q = 0.0031
            # lr_mu = 0.009
            # agent_dual_q = LoggingDualQ(
            #     env,
            #     lr_q = lr_q,
            #     lr_mu = lr_mu,
            #     log_every = 1000,
            #     seed = ALG_SEED
            # )
            # for t in range(1, n):
            #     agent_dual_q.step(t)
            

            agents = [agent_canonical, agent_dual, agent_qreps]
            agent_names = ['Canonical Primal-Dual', 'Primal-Dual API', 'Q-REPS']


            fig, axs = plt.subplots(2, len(agents), figsize=(20, 10), gridspec_kw={'height_ratios': [1, 1.5]})

            def plot_policy_with_heatmap(ax, policy, value_function, env_image=None):
                # Compute U and V components for the quiver plot
                U = (policy[:, 2] - policy[:, 0]).reshape(M, M)
                V = (policy[:, 3] - policy[:, 1]).reshape(M, M)
                
                # Reshape the value function to match the grid
                heatmap_values = value_function.reshape(M, M)
                
                # Create the meshgrid for plotting
                x = np.arange(M) + 0.5
                X, Y = np.meshgrid(x, x[::-1])

                # Plot the heatmap
                cax = ax.imshow(heatmap_values, cmap='gray', extent=[0, M, 0, M])
                
                # Add a colorbar below the plot
                divider = make_axes_locatable(ax)
                cax_cb = divider.append_axes("bottom", size="5%", pad=0.2)
                fig.colorbar(cax, cax=cax_cb, orientation='horizontal')

                # Overlay the environment image if provided
                ax.imshow(env_image, extent=[0, M, 0, M], alpha=0.5)

                # Plot the policy using quiver
                ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=2, color='red')

                # Set axis limits and aspect ratio
                ax.set(xlim=[0, M], ylim=[0, M], aspect='equal')

            for i, agent in enumerate(agents):
                # Plot rewards with smoothing
                df = agent.df
                rewards = agent._rewards
                # smoothed = pd.Series(rewards).rolling(window=10, center=False).mean()
                if hasattr(agent, "N"):
                    axs[0, i].plot(range(0, len(agent._rewards) * 100 * agent.N, 100*agent.N), rewards, color="blue")
                else:
                    axs[0, i].plot(range(0, len(agent._rewards) * 100, 100), rewards, color="blue")
                # axs[0, i].plot(range(0, len(smoothed) * 1000, 1000), smoothed, color="blue")
                axs[0, i].grid()
                axs[0, i].set_xlabel("Step")
                axs[0, 0].set_ylabel("Expected Discounted Return")
                axs[0, i].set_title(f"{agent_names[i]}", fontsize=14)
                axs[0, i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axs[0, i].ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
                
                # Plot policy with heatmap
                if hasattr(agent, "q"):
                    v = np.sum(agent.policy * agent.q, axis=1)
                else:
                    v = agent.v
                plot_policy_with_heatmap(axs[1, i], agent.policy, v, env_image=env_image)

            fig.tight_layout()
            plt.savefig(f'dual_algorithm_comparison_{RANDOM_SEED}_stochastic_8x8_example_hard.png')
            plt.close()
            # print(results)
            # print("End time: ", time.time() - start)

            # save_results(results, row['index'])
