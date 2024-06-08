import argparse 
from copy import deepcopy
import random
import time
import gymnasium as gym
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from ray import train
from ..utils import make_env, QNetwork, QREPSPolicy
import logging

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 1

def tune_elbe(config):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    args = argparse.Namespace(**config)
    if "__trial_index__" in config: args.seed = config["__trial_index__"] + SEED_OFFSET
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    args.minibatch_size = args.total_iterations // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.total_iterations
    args.num_steps = args.total_iterations // args.num_envs

    logging_callback=lambda r: train.report({'reward':r})

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
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

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
    mean_rewards = []
    full_rewards = []
    if args.save_learning_curve: rewards_df = pd.DataFrame(columns=["Step", "Reward"])

    for iteration in range(1, args.num_iterations + 1):
        reward_iteration = []
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
                        mean_rewards.append(info["episode"]["r"])
                        reward_iteration.append(info["episode"]["r"])
                        print(f'Iteration: {global_step}, Reward: {info["episode"]["r"]}')
                
        if args.save_learning_curve and len(reward_iteration) > 0: 
            rewards_df = rewards_df._append({"Step": global_step, "Reward": np.mean(reward_iteration)}, ignore_index=True)
            print(f'Iteration: {global_step}, Reward: {np.mean(reward_iteration)}')
        if len(reward_iteration) >0: full_rewards.append(np.mean(reward_iteration))
        
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

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.total_iterations, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    q, values = qf.get_values(b_obs[mb_inds], b_actions[mb_inds], actor)
                    values_next = qf.get_values(b_next_obs[mb_inds], b_actions[mb_inds], actor)[1]

                    if args.target_network:
                        delta = b_rewards[mb_inds].squeeze() + args.gamma * qf_target.get_values(b_next_obs[mb_inds], policy=actor)[1].detach() * (1 - b_dones[mb_inds].squeeze()) - q        
                        
                    elif args.gae:
                        delta = b_returns[mb_inds] - q
                    
                    else: delta = b_rewards[mb_inds].squeeze() + args.gamma * values_next * (1 - b_dones[mb_inds].squeeze()) - q

                    critic_loss = eta * torch.log(torch.mean(torch.exp(delta / eta), 0)) + torch.mean((1 - args.gamma) * values, 0)

                    q_optimizer.zero_grad()
                    critic_loss.backward()
                    q_optimizer.step()

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
        return rewards_df
