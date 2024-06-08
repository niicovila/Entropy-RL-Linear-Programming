import os
import sys
import ray
from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from ..algorithms.qreps import tune_elbe, tune_saddle



config_ray = {
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
    "num_envs": tune.choice([16, 32, 64, 128, 256]),
    "gamma": 0.99,

    "total_iterations": tune.choice([512, 1024, 2048, 4096]),
    "num_minibatches": tune.choice([4, 8, 16, 32, 64]),
    "update_epochs": tune.choice([10, 25, 50, 100, 150, 300]),
    "update_epochs_policy": 50,

    "alpha": tune.choice([2, 4, 8, 12, 32, 64, 100]),  
    "eta": tune.choice([2, 4, 8, 12, 32, 64, 100]),

    # Learning rates
    "beta": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "policy_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "q_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "anneal_lr": tune.choice([True, False]),

    # Layer Init
    "layer_init": tune.choice(["default", 
                               "orthogonal_gain", 
                               "orthogonal", 
                               "xavier_normal", 
                               "xavier_uniform", 
                               "kaiming_normal", 
                               "kaiming_uniform", 
                               "sparse"]),
    # Architecture
    "policy_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "num_hidden_layers": 2,
    "hidden_size": tune.choice([16, 32, 64, 128, 256, 512]),
    "actor_last_layer_std": 0.01,

    "q_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "q_num_hidden_layers": 2,
    "q_hidden_size": tune.choice([16, 32, 64, 128, 256, 512]),
    "q_last_layer_std": 1.0,

    "average_critics": tune.choice([True, False]),
    "use_policy": tune.choice([True, False]),

    "parametrized_sampler" : tune.choice([True, False]),
    "sampler_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "sampler_num_hidden_layers": 2,
    "sampler_hidden_size": tune.choice([16, 32, 64, 128, 256, 512]),
    "sampler_last_layer_std": 0.1,

    # Optimization
    "q_optimizer": "Adam",  # "Adam", "SGD", "RMSprop
    "actor_optimizer": "Adam", 
    "sampler_optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "normalize_delta": tune.choice([True, False]),
    "gae": tune.choice([True, False]),
    "gae_lambda": 0.95,
    "use_kl_loss": tune.choice([True, False]),
    "q_histogram": False,

    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0, 
    "save_learning_curve": False,
    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}

config_elbe = {
    "exp_name": "QREPS",
    "seed": 1,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,

    "env_id": "LunarLander-v2",

    # Algorithm
    "total_timesteps": 100000,
    "num_envs": tune.choice([16, 32, 64, 128, 256]),
    "gamma": tune.choice([0.95, 0.97, 0.99, 0.999]),

    "total_iterations": tune.choice([512, 1024, 2048]),
    "num_minibatches": tune.choice([4, 8, 16, 32, 64]),
    "update_epochs": tune.choice([10, 25, 50, 100, 150, 300]),
    "update_epochs_policy": 50,

    "alpha": tune.choice([2, 4, 8, 12, 32, 64, 100]),  
    "eta": None,  

    # Learning rates
    "policy_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "q_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "anneal_lr": tune.choice([True, False]),

    # Layer Init
    "layer_init": tune.choice(["default", 
                               "orthogonal_gain", 
                               "orthogonal", 
                               "xavier_normal", 
                               "xavier_uniform", 
                               "kaiming_normal", 
                               "kaiming_uniform", 
                               "sparse"]),
    # Architecture
    "policy_activation": "Tanh",
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "actor_last_layer_std": 0.01,

    "q_activation": "Tanh",
    "q_num_hidden_layers": 4,
    "q_hidden_size": 128,
    "q_last_layer_std": 1.0,
    "use_policy": tune.choice([True, False]),

    # Optimization
    "q_optimizer": "Adam",  # "Adam", "SGD", "RMSprop
    "actor_optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "average_critics": tune.choice([True, False]),
    "normalize_delta": tune.choice([True, False]),
    "use_kl_loss": tune.choice([True, False]),
    "q_histogram": False,

    "target_network": False,
    "gae": tune.choice([True, False]),
    "gae_lambda": 0.95,
    "save_learning_curve": False,

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}



if "__main__" == __name__:

    num_cpus = int(sys.argv[1])
    ray.init(address=os.environ['ip_head'])
    num_cpus = 3
    current_dir = os.getcwd()
    num_samples = 512
    seeds = 3

    search_alg = HEBOSearch(metric="reward", mode="max")
    search_alg = Repeater(search_alg, repeat=seeds)
    config = config_ray
    run = 'LunarLander-v2-Saddle'

    tuner = tune.Tuner(  
        tune.with_resources(tune_saddle, resources={"cpu": 1}),
        tune_config=tune.TuneConfig(
            metric="reward",
            mode="max",
            search_alg=search_alg,
            scheduler=None,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name=f"qreps-tuning-{run}",
            storage_path=current_dir + "/ray_results", 
        ),
        param_space=config,
    )

    result_grid = tuner.fit()
    print("Best config is:", result_grid.get_best_result().config)
    results_df = result_grid.get_dataframe()
    results_df.to_csv(f"tune_results_{run}_{num_samples}.csv")

