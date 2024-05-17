#!/bin/bash
#SBATCH --job-name=qreps-tune
#SBATCH -p high

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=2

### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5g # Memory per node. 5 GB (In total, 10 GB)
#SBATCH --cpus-per-task=6
#SBATCH --output=qreps_tune_job_gpu.out
#SBATCH --error=qreps_tune_job_gpu.err

# Download the source repository
[ ! -e "nccl-tests" ] && git clone https://github.com/NVIDIA/nccl-tests.git

# Load module libraries
module load foss/2020b
module load NCCL/2.8.3-GCCcore-10.2.0-CUDA-11.4.3
module load "Miniconda3/4.9.2"

eval "$(conda shell.bash hook)"
conda activate qreps
poetry install
poetry install -E atari
set -x

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" /bin/hostname --ip-address)

echo "$head_node"
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.

if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6392
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    $(which ray) start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        $(which ray) start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

poetry run python -u ../tuning/tune.py "$SLURM_CPUS_PER_TASK"