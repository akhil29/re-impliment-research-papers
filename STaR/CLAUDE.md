# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Paper Summary

**STaR: Self-Taught Reasoner - Bootstrapping Reasoning With Reasoning** (NeurIPS 2022)

STaR iteratively improves a language model's reasoning by:
1. Generating chain-of-thought rationales for problems using few-shot prompts
2. Filtering to keep only rationales that lead to correct answers
3. For incorrect answers, using "rationalization" - re-prompting with the answer as a hint
4. Fine-tuning on successful rationales, then repeating

Key insight: The model learns from its own generated reasoning, bootstrapping from a small set of example rationales.

## Algorithm (from paper/sections/method.tex)

```
Input: pretrained LLM M, dataset D = {(x_i, y_i)}, few-shot prompts P
For each iteration n:
  1. Generate rationales: (r_i, y_hat_i) = M(x_i) for all problems
  2. Rationalize failures: (r_rat_i, y_rat_i) = M(x_i + hint) where y_hat != y
  3. Filter correct: D_n = {(x_i, r_i, y_i) | y_hat_i == y_i}
  4. Filter rationalized: D_rat_n = {(x_i, r_rat_i, y_i) | y_hat != y AND y_rat == y}
  5. Fine-tune: M_n = train(M_original, D_n + D_rat_n)
```

## Running the STaR Loop

```bash
# Main command - orchestrates the full iterative training
python3 iteration_train.py --task commonsenseqa --rationalize --n_iters 10

# Key flags:
#   --rationalize         Enable rationalization for wrong examples
#   --task                Dataset: commonsenseqa, gsm, arithmetic
#   --n_iters             Number of outer loop iterations
#   --base_model_location GCS path to GPT-J checkpoint
#   --dry_run             Test without actual training
```

### Training Schedule Options

```bash
# Epoch-based (default): scales with dataset size
python3 iteration_train.py --base_epochs 1.0 --add_epochs 0.2

# Step-based linear growth
python3 iteration_train.py --steady_grow --start_steps 40 --add_steps 20

# Step-based exponential growth
python3 iteration_train.py --exponential_grow --start_steps 40 --grow_steps 1.2
```

## Core Components

| Script | Purpose |
|--------|---------|
| `iteration_train.py` | Main loop: coordinates generation, filtering, training |
| `device_inference.py` | Generates rationales from current model (runs on TPU) |
| `device_train.py` | Fine-tunes model on filtered dataset (runs on TPU) |
| `create_finetune_tfrecords.py` | Converts text outputs to TFRecords for training |

### Data Flow

```
iteration_train.py
    |
    v
device_inference.py (generate rationales)
    |
    v
{task}/{experiment}/{experiment}_N.txt (raw generated text)
    |
    v
create_finetune_tfrecords.py
    |
    v
{task}/{experiment}/{experiment}_N.tfrecords
    |
    v
device_train.py (fine-tune)
    |
    v
gs://{bucket}/{model_dir}/step_N/ (checkpoint)
```

## Datasets

| Dataset | Task | Prompts Location |
|---------|------|------------------|
| CommonsenseQA | Multiple-choice reasoning | `commonsenseqa/prompts.txt` |
| GSM8K | Grade school math | `gsm/prompts.txt` |
| Arithmetic | Multi-digit addition with scratchpad | `arithmetic/` |

Rationalization prompts (with answer hints): `{task}/prompts_answer_key.txt`

## Key Hyperparameters (from paper)

- **Base model**: GPT-J 6B
- **Initial training steps**: 40 (arithmetic: 300 with rationalization)
- **Step increase per iteration**: +20%
- **Learning rate warmup**: 100 steps
- **Sampling temperature**: Low (greedy-ish) - high temp causes bad rationales
- **p_rationalization**: Fraction of failures to rationalize (default: 1.0)

## Infrastructure Requirements

- TPU v3-8 for `device_*.py` scripts
- Google Cloud Storage bucket for checkpoints
- JAX 0.2.12 (critical for v1 models)

```bash
pip install -r requirements.txt
pip install jax==0.2.12  # Must match this version
```

## Paper Results

| Dataset | Few-shot | Fine-tuned | STaR | STaR + Rat |
|---------|----------|------------|------|------------|
| CommonsenseQA | 36.6% | 60.0% | 68.8% | **72.5%** |
| GSM8K | 3.1% | 5.8% | 10.1% | **10.7%** |
| Arithmetic | ~0% | 76.3% | - | **89.5%** |

## Paper Source

The paper LaTeX source is in `paper/`. Main file: `paper/main.tex`
