from copy import deepcopy
import json
import os
import random
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

def layer_init(layer, args, gain_ort=np.sqrt(2), bias_const=0.0, gain=1):
    if args.layer_init == "orthogonal_gain":
        torch.nn.init.orthogonal_(layer.weight, gain_ort)
    elif args.layer_init == "orthogonal":
        torch.nn.init.orthogonal_(layer.weight, gain)
    elif args.layer_init == "xavier_normal":
        torch.nn.init.xavier_normal_(layer.weight, gain)
    elif args.layer_init == "xavier_uniform":
        torch.nn.init.xavier_uniform_(layer.weight, gain)
    elif args.layer_init == "kaiming_normal":
        torch.nn.init.kaiming_normal_(layer.weight)
    elif args.layer_init == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(layer.weight)
    elif args.layer_init == "sparse":
        torch.nn.init.sparse_(layer.weight, sparsity=0.1)
    else:
        pass
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

class Sampler(nn.Module):
    def __init__(self, args, N):
        super().__init__()
        self.n = N
            
        self.z = nn.Sequential(
            (nn.Linear(N, args.sampler_hidden_size)),
            getattr(nn, args.sampler_activation)(),
            *[layer for _ in range(args.sampler_num_hidden_layers) for layer in (
                (nn.Linear(args.sampler_hidden_size, args.sampler_hidden_size)),
                getattr(nn, args.sampler_activation)()
            )],
            (nn.Linear(args.sampler_hidden_size, N)),
        )

    def forward(self, x):
        return self.z(x)

    def get_probs(self, x):
        logits = self(x)
        sampler_dist = Categorical(logits=logits)
        return sampler_dist.probs

class ExponentiatedGradientSampler:
    def __init__(self, N, device, eta, beta=0.01):
        self.n = N
        self.eta = eta
        self.beta = beta

        self.h = torch.ones((self.n,)) / N
        self.z = torch.ones((self.n,)) / N

        self.prob_dist = Categorical(torch.ones((self.n,))/ N)
        self.device = device

    def reset(self):
        self.h = torch.ones((self.n,))
        self.z = torch.ones((self.n,))
        self.prob_dist = Categorical(torch.softmax(torch.ones((self.n,)), 0))
                                     
    def probs(self):
        return self.prob_dist.probs.to(self.device)
    
    def entropy(self):
        return self.prob_dist.entropy().to(self.device)
    
    def update(self, bellman):
        self.h = bellman -  self.eta * torch.log(self.n * self.probs())
        t = self.beta*self.h
        t = torch.clamp(t, -50, 50)
        self.z = self.probs() * torch.exp(t)
        self.z = torch.clamp(self.z / (torch.sum(self.z)), min=1e-8, max=1.0)
        self.prob_dist = Categorical(self.z)

class QNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.alpha = args.alpha
        self.use_policy = args.use_policy

        def init_layer(layer, gain_ort=np.sqrt(2), gain=1):
            if args.layer_init == "default":
                return layer
            else:
                return layer_init(layer, args, gain_ort=gain_ort, gain=gain)

        self.critic = nn.Sequential(
            init_layer(nn.Linear(np.array(env.single_observation_space.shape).prod(), args.q_hidden_size)),
            getattr(nn, args.q_activation)(),
            *[layer for _ in range(args.q_num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.q_hidden_size, args.q_hidden_size)),
                getattr(nn, args.q_activation)()
            )],
            init_layer(nn.Linear(args.q_hidden_size, env.single_action_space.n), gain_ort=args.q_last_layer_std, gain=args.q_last_layer_std),
        )

    def forward(self, x):
        return self.critic(x)
    
    def get_values(self, x, action=None, policy=None):
        q = self(x)
        z = q / self.alpha
        if self.use_policy:
            if policy is None: pi_k = torch.ones(x.shape[0], self.env.single_action_space.n, device=x.device) / self.env.single_action_space.n
            else: _, _, _, pi_k = policy.get_action(x); pi_k = pi_k.detach()
            v = self.alpha * (torch.log(torch.sum(pi_k * torch.exp(z), dim=1))).squeeze(-1)
        else:
            v = self.alpha * torch.log(torch.mean(torch.exp(z), dim=1)).squeeze(-1)
        if action is None:
            return q, v
        else:
            q = q.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)
            return q, v
    
    
class QREPSPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        def init_layer(layer, gain_ort=np.sqrt(2), gain=1):
            if args.layer_init == "default":
                return layer
            else:
                return layer_init(layer, args, gain_ort=gain_ort, gain=gain)

        self.actor = nn.Sequential(
            init_layer(nn.Linear(np.array(env.single_observation_space.shape).prod(), args.hidden_size)),
            getattr(nn, args.policy_activation)(),
            *[layer for _ in range(args.num_hidden_layers) for layer in (
                init_layer(nn.Linear(args.hidden_size, args.hidden_size)),
                getattr(nn, args.policy_activation)()
            )],
            init_layer(nn.Linear(args.hidden_size, env.single_action_space.n), gain_ort=args.actor_last_layer_std, gain=args.actor_last_layer_std),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x, action=None):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        if action is None: action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        action_log_prob = policy_dist.log_prob(action)
        return action, action_log_prob, log_prob, action_probs
    
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CartPole-QREPS-Elbe-Benchmark"
    """the wandb's project name"""
    wandb_entity = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_learning_curve: bool = False
    """whether to save the learning curve results"""

    # Dynamics settings
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel game environments"""
    total_iterations: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    update_epochs: int = 100
    """the number of epochs for the policy and value networks"""
    num_minibatches: int = 8
    """the number of minibatches to train the policy and value networks"""

    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 2.5e-4
    """the learning rate of the Q network network optimizer"""
    beta: float = 0.01
    """the beta coefficient of the sampler network"""

    # Regularization
    alpha: float = 2.0
    """Entropy regularization coefficient."""
    eta = None
    """coefficient for the kl reg"""

    # Policy Network params
    policy_activation: str = "Tanh"
    """the activation function of the policy network"""
    hidden_size: int = 128
    """the hidden size of the policy network"""
    num_hidden_layers: int = 2
    """the number of hidden layers of the policy network"""
    actor_last_layer_std: float = 0.01
    """the standard deviation of the last layer of the Q network"""

    # Q Network params
    q_activation: str = "Tanh"
    """the activation function of the Q network"""
    q_hidden_size: int = 128
    """the hidden size of the Q network"""
    q_num_hidden_layers: int = 4
    """the number of hidden layers of the Q network"""
    q_last_layer_std: float = 1.0
    """the standard deviation of the last layer of the Q network"""

    # Sampler
    parametrized_sampler: bool = False
    """if toggled, the sampler will be parametrized"""
    sampler_hidden_size: int = 128
    """the hidden size of the sampler network"""
    sampler_activation: str = "Tanh"
    """the activation function of the sampler network"""
    sampler_num_hidden_layers: int = 2
    """the number of hidden layers of the sampler network"""

    # Optimizer params
    q_optimizer: str = "Adam"
    """the optimizer of the Q network"""
    actor_optimizer: str = "Adam"
    """the optimizer of the policy network"""
    sampler_optimizer: str = "Adam"
    """the optimizer of the sampler network"""
    eps: float = 1e-8
    """the epsilon value for the optimizer"""

    # Options
    anneal_lr: bool = True
    """if toggled, the learning rate will decrease linearly"""
    normalize_delta: bool = False
    """if toggled, the delta will be normalized"""
    layer_init: str = "xavier_uniform"
    """the initialization method of the layers"""
    use_policy: bool = True
    """if toggled, the policy will be used in the Q function"""
    gae: bool = False
    """if toggled, the generalized advantage estimation will be used"""
    gae_lambda: float = 0.95
    """the lambda coefficient of the generalized advantage estimation"""
    target_network: bool = False
    """if toggled, the target network will be used"""
    target_network_frequency: int = 100
    """the frequency of updating the target network"""
    average_critics: bool = False
    """if toggled, the critics will be averaged"""
    use_kl_loss: bool = False
    """if toggled, the kl loss will be used"""
    
    # to be filled in runtime
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    minibatch_size: int = 0
    """the minibatch size (computed in runtime)"""
    num_steps: int = 0
    """the number of steps (computed in runtime)"""

def load_args_from_json(filepath: str):
    # Read the JSON string from the file
    with open(filepath, 'r') as f:
        json_string = f.read()
    args_dict = json.loads(json_string)
    return Args(**args_dict)

if __name__ == "__main__":
    start_time = time.time()
    config_filepath = '../configs/config_saddle.json'
    args = load_args_from_json(config_filepath)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.minibatch_size = args.total_iterations // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.total_iterations
    args.num_steps = args.total_iterations // args.num_envs

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    actor = QREPSPolicy(envs, args).to(device)
    qf = QNetwork(envs, args).to(device)

    if args.target_network:
        qf_target = QNetwork(envs, args).to(device)
        qf_target.load_state_dict(qf.state_dict())

    if args.q_optimizer == "Adam" or args.q_optimizer == "RMSprop":
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr, eps=args.eps
        )
    else:
        q_optimizer = getattr(optim, args.q_optimizer)(
            list(qf.parameters()), lr=args.q_lr
        )
    if args.actor_optimizer == "Adam" or args.actor_optimizer == "RMSprop":
        actor_optimizer = getattr(optim, args.actor_optimizer)(
            list(actor.parameters()), lr=args.policy_lr, eps=args.eps
        )
    else:
        actor_optimizer = getattr(optim, args.actor_optimizer)(
            list(actor.parameters()), lr=args.policy_lr
        )
    alpha = args.alpha
    if args.eta is None: eta = args.alpha
    else: eta = torch.Tensor([args.eta]).to(device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    next_observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # Added this line
    values_hist = torch.zeros((args.num_steps, args.num_envs)).to(device)
    qs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    reward_iteration = []
    full_rewards = []

    if args.save_learning_curve: rewards_df = pd.DataFrame(columns=["Step", "Reward"])
    for iteration in range(1, args.num_iterations + 1):
        mean_reward = []

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.policy_lr
            actor_optimizer.param_groups[0]["lr"] = lrnow

            lrnow = frac * args.q_lr
            q_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            
            # ALGO LOGIC: action logic
            with torch.no_grad():        
                action, _, logprob, _ = actor.get_action(next_obs)
                if args.gae:
                    q, value = qf.get_values(next_obs, action)
                    qs[step] = q
                    values_hist[step] = value

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            reward = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_observations[step] = next_obs
            dones[step] = next_done

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        reward_iteration.append(info["episode"]["r"])
                        mean_reward.append(info["episode"]["r"])
                        print(f"Step: {global_step}, Episode: {len(reward_iteration)}, Reward: {info['episode']['r']}")

        if args.save_learning_curve and len(mean_reward) > 0: 
            rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(mean_reward)}, ignore_index=True)

        if args.gae:
            # bootstrap value if not done
            with torch.no_grad():
                next_value = qf.get_values(next_obs)[1]
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values_hist[t + 1] if t + 1 < args.num_steps else next_value
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - qs[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + qs

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape((-1, envs.single_action_space.n))
        b_rewards = rewards.flatten()
        b_dones = dones.flatten()
        b_inds = np.arange(args.total_iterations)
        if args.gae:
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

        weights_after_each_epoch = []

        np.random.shuffle(b_inds)
        for start in range(0, args.total_iterations, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            if args.parametrized_sampler:
                sampler = Sampler(args, N=args.minibatch_size).to(device)
                sampler_optimizer = getattr(optim, "Adam")(
                        list(sampler.parameters()), lr=args.beta, eps=args.eps
                    )
            else:
                sampler = ExponentiatedGradientSampler(args.minibatch_size, device, eta, args.beta)

            for epoch in range(args.update_epochs):            
                q, values = qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)
                values_next = qf.get_values(b_next_obs[mb_inds], b_actions[mb_inds], actor)[1]

                if args.target_network:
                    delta = b_rewards[mb_inds].squeeze() + args.gamma * qf_target.get_values(b_next_obs[mb_inds], policy=actor)[1].detach() * (1 - b_dones[mb_inds].squeeze()) - q        
                
                elif args.gae:
                    delta = b_returns[mb_inds] - q

                else: delta = b_rewards[mb_inds].squeeze() + args.gamma * values_next * (1 - b_dones[mb_inds].squeeze()) - q
                if args.normalize_delta: delta = (delta - delta.mean()) / (delta.std() + 1e-9)

                bellman = delta.detach()
                if args.parametrized_sampler: z_n = sampler.get_probs(bellman) 
                else: z_n = sampler.probs() 
                
                critic_loss = torch.sum(z_n.detach() * (delta - eta * torch.log(sampler.n * z_n.detach()))) + (1 - args.gamma) * values.mean()
            
                q_optimizer.zero_grad()
                critic_loss.backward()
                q_optimizer.step()

                if args.parametrized_sampler:
                    sampler_loss = - (torch.sum(z_n * (bellman - eta * torch.log(sampler.n * z_n))) + (1 - args.gamma) * values.mean().detach())
                    
                    sampler_optimizer.zero_grad()
                    sampler_loss.backward()
                    sampler_optimizer.step()

                else: sampler.update(bellman)


                if args.use_kl_loss: 
                    with torch.no_grad():
                        q_state_action, val = qf.get_values(b_obs[mb_inds], policy=actor)
                    _, _, newlogprob, probs = actor.get_action(b_obs[mb_inds])

                    if args.gae:
                        advantadge = q_state_action - b_returns[mb_inds].unsqueeze(1)
                    else:     
                        advantadge = q_state_action - val.unsqueeze(1)
                    
                    if args.normalize_delta: advantadge = (advantadge - advantadge.mean()) / (advantadge.std() + 1e-8)
                    actor_loss = torch.mean(probs * (alpha * (newlogprob-b_logprobs[mb_inds].detach()) - advantadge))

                else:
                    if args.gae:
                        if args.normalize_delta: advantadge = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                        else: advantadge = b_advantages[mb_inds]
                    else:
                        q, values = qf.get_values(b_obs[mb_inds], b_actions[mb_inds])
                        advantadge = q - values
                        if args.normalize_delta: advantadge = (advantadge - advantadge.mean()) / (advantadge.std() + 1e-8)

                    weights = torch.clamp(advantadge / alpha, -50, 50)
                    _, log_likes, _, _ = actor.get_action(b_obs[mb_inds], b_actions[mb_inds])
                    actor_loss = -torch.mean(torch.exp(weights.detach()) * log_likes)
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            if args.average_critics: weights_after_each_epoch.append(deepcopy(qf.state_dict()))
        
        if args.average_critics:
            avg_weights = {}
            for key in weights_after_each_epoch[0].keys():
                avg_weights[key] = sum(T[key] for T in weights_after_each_epoch) / len(weights_after_each_epoch)
            qf.load_state_dict(avg_weights)

        if args.target_network and iteration % args.target_network_frequency == 0:
            for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    envs.close()
    writer.close()
    if args.save_learning_curve: 
        rewards_df.to_csv(f"./runs/{run_name}/rewards.csv", index=False)