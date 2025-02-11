# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
model_type = "gpt"

wandb_log = True
wandb_project = "tmlt"
wandb_run_name = "gpt2-44.64M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12 * 4
block_size = 1024
gradient_accumulation_steps = 1

n_layer = 6
n_head = 8
n_embd = 512
