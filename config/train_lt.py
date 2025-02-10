model_type = "lt"

wandb_log = True
wandb_project = "tmlt"
# wandb_run_name = "lt-65.1M"  # "lt-155.5M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 4

n_loop = 6
n_head = 16
n_embd = 1024  # 2048

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

wandb_run_name = f"lt-65.1M-{n_loop}"
