#!/bin/bash

### ATTENTION: this script is provided for reference only to give an idea how to make it run on SLURM-based systems
### you need to adjust the script to your needs and test it properly

### START: do not change this
#SBATCH --account=<your_project>
#SBATCH --licenses=<your_licenses>
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=<your_constraints>
#SBATCH --gpus-per-node=<your_gpus_per_node>
### END: do not change this

### START: usually you do not need to change this
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=<your_qos>
### END: usually you do not need to change this

### START: feel free to change this
#SBATCH --job-name=<your_job_name>
#SBATCH --time=<your_time>
#SBATCH --nodes=<your_nodes>
### END: feel free to change this

### START: do not change this
set -x -e

module load cudatoolkit
module load cray-mpich
module load gcc
module load conda
conda activate <conda environment>

# set all temporary folders to /tmp on the compute node
export RUN_ROOT="/tmp"
mkdir -p "$RUN_ROOT"/{hf,xdg,tmp,pycache,triton}
export HF_HOME="$RUN_ROOT/hf"
export HF_DATASETS_CACHE="$RUN_ROOT/hf/datasets"
export HF_HUB_CACHE="$RUN_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$RUN_ROOT/hf/transformers"
export XDG_CACHE_HOME="$RUN_ROOT/xdg"
export TMPDIR="$RUN_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache"
export TRITON_CACHE_DIR="$RUN_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/torchinductor"

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export WORLD_SIZE=$(($GPUS_PER_NODE*$SLURM_NNODES))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
### END: do not change this

## add this to resume from checkpoint
## --resume_from_checkpoint checkpoint_path \
## --ignore_data_skip True \
## ignore data skip until https://github.com/huggingface/transformers/pull/33544 is merged

srun --kill-on-bad-exit=1 bash -c '
  export WORLD_SIZE=$SLURM_NTASKS
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  echo "host=$(hostname) RANK=$RANK CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>} MASTER=$MASTER_ADDR:$MASTER_PORT"

  exec python -u \
 $PSCRATCH/netFound/src/netFoundPretraining.py \
 --report_to tensorboard \
 --bf16 \
 --do_train \
 --do_eval \
 --include_num_input_tokens_seen True \
 --include_tokens_per_second True \
 --eval_strategy steps \
 --save_strategy steps \
 --dataloader_num_workers 16 \
 --dataloader_persistent_workers False \
 --dataloader_prefetch_factor 8 \
 --streaming True \
 --gradient_accumulation_steps 1 \
 --ignore_data_skip True \
 --size base \
 --per_device_train_batch_size 20 \
 --per_device_eval_batch_size 20 \
 --train_dir /train/data/location \
 --test_dir /test/data/location \
 --output_dir /output/data/location \
 --learning_rate 0.00005 \
 --lr_scheduler_type warmup_stable_decay \
 --warmup_steps 500 \
 --metadata_loss_weight 0.4 \
 --max_grad_norm 0.5 \
 --lr_scheduler_kwargs "{\"num_decay_steps\": 500}" \
 --logging_steps 5000 \
 --save_steps 5000 \
 --seed 69 \
 --max_steps 45000000
 '

# expected performance: ~1it/sec for base model on 32 nodes with 4xA100 40GB per node over infiniband
# additional flags:
# sizes: small, base, large
# --group_by_length 
# --use_flash_attn False
