model_type = "tmlt"

wandb_log = True
wandb_project = "tmlt"
# wandb_run_name = "tmlt-70.7M"  # 177.0M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12 * 4
block_size = 1024
gradient_accumulation_steps = 1

n_loop = 48
n_head = 12
n_embd = 768

wandb_run_name = f"tmlt-49.6M-{n_loop}"
