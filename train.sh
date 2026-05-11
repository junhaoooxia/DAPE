#!/bin/bash
# DAPE Two-Stage Training
# Stage 1: Adjustable Norm-tuning (400 steps, lr=5e-5)
# Stage 2: Visual Adapter tuning (70 steps, lr=1e-5) -- triggered automatically

python main.py \
  -b configs/template.yaml \
  --wandb False \
  --train True
