# Timestep-Modulated Looped Transformer (TMLT)

This repository is the official implementation of [On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](https://arxiv.org/abs/2410.01405).

### Install

```shell
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Training WikiText-103

Train a standard GPT model following [nanoGPT](https://github.com/karpathy/nanoGPT):
```bash
python train.py config/train_gpt2.py
```

Train a Looped Transformer with or without Timestep Encoding:
```bash
python train.py config/train_lt.py
python train.py config/train_tmlt.py
```

### Acknowledgement

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [DiT](https://github.com/facebookresearch/DiT)