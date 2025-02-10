# Timestep-Modulated Looped Transformer (TMLT)

This repository is the official implementation of [On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](https://arxiv.org/abs/2410.01405).

## Setup

```shell
conda env create -f environment.yml
conda activate timestep
```

## Experiments

### Dynamic Programming

```bash
cd CoT_benchmark

# Generate the data
python3 tasks/ED/data.py --file "/data/ED/60_24" --length 60 --train_size 1e6 --test_size 1e3 --using 24 # Here `using` + 2 = the max size of working vocabulary.
python3 tasks/LIS/data.py --file 
python3 tasks/CYK/data.py --file 

# Train the model
num_loop=100
model_name="LoopedGPT" # TimeDependentLoopedGPT
task="ED" # LIS, CYK

torchrun --standalone --nproc_per_node=2 train.py --file "data/${task}" --folder "tasks/${task}" --output_dir "output/${task}/${model_name}_${num_loop}" \
 --wandb_name "ED_60_${model_name}_${num_loop}" --model "$model_name" --maxlen 127 --maxdata 127 --vocab 211 --num_range 180 --learning_rate 1e-4 \
 --weight_decay 0.01 --drop 0.0 --batch_size 64 --epoch 100 --warmup 5 --dmodel 256 --head 4 --num_layer 1 --num_loop "$num_loop"
```

### Language Modeling: WikiText-103

```bash
cd nanoGPT
torchrun --standalone --nproc_per_node=2 train.py config/train_looped.py
torchrun --standalone --nproc_per_node=2 train.py config/train_time_dependent.py
torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2.py
```
You can control parameters in `config/train_looped.py` and `config/train_time_dependent.py`.

### In-Context Learning

```bash
cd in_context

n_gpu=0, b=50, T=50
python scripts/train.py --config configs/decision_tree/base_loop.yaml \
    --model.n_layer 1 \
    --training.curriculum.loops.start $T \
    --training.curriculum.loops.end $b \
    --training.n_loop_window $T \
    --wandb.name "DT_loop_L1_ends{$b}_T{$T}" \
    --gpu.n_gpu $n_gpu
```
Use `configs/decision_tree/base_time.yaml` for timestep encoding models.

## Acknowledgement

- [Code for "Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective"](https://github.com/guyuntian/CoT_benchmark)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Code for "Looped Transformers are Better at Learning Learning Algorithms"](https://github.com/Leiay/looped_transformer)
