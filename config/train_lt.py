model_type = "lt"

wandb_log = True
wandb_project = "tmlt"
# wandb_run_name = "lt-65.1M"  # "lt-155.5M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12 * 4
block_size = 1024
gradient_accumulation_steps = 1

n_loop = 24
n_head = 12
n_embd = 768

wandb_run_name = f"lt-46.5M-{n_loop}"
