# MAPNet

This repository contains code for the paper ***Mucus-penetrability guided AI enables de novo design of inhalable antimicrobials against pneumonia***

## Environment Setup

Create the environment from the provided conda file:

```bash
conda env create -f config/conda_env.yml
conda activate MAPNet
```

## Model Files

we provide the trained MAPNet for direct usage.
[Download model weights:](https://drive.google.com/file/d/1gbAJWpSyh1Iy-jVLHpTOmvdY3RlIlqr-/view?usp=sharing)

Please place the checkpoint in `checkpoints/model.pt`.

## Candidate Generation

Run peptide generation with the default config and checkpoint path:

```bash
python design.py
```

Common options:

```bash
python design.py \
  --config config/config.yaml \
  --model_path checkpoints/model.pt \
  --n_candidates 10 \
  --seq_length 9 \
  --n_mcmc_runs 5 \
  --mcmc_iterations 200 \
  --output generated_peptides.txt
```

Use fixed amino acid positions with `--fixed_positions`:

```bash
python design.py --fixed_positions "0:M,3:C,7:K"
```

## Outputs

Generated candidates are saved as a tab-separated file:

```text
Rank    Sequence    Predicted_Activity
```
