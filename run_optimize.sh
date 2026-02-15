#!/bin/sh

python -m optimize \
  --opt-model Qwen/Qwen2.5-0.5B-Instruct \
  --opt-temp 0.7 \
  --eval-model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-temp 0.3 \
  --exec-model Qwen/Qwen2.5-0.5B-Instruct \
  --exec-temp 0 \
  --workspace "workspace" \
  --initial-round 1 \
  --max-rounds 3 \
  --template "Poem.yaml" \
  --name "poem_test" \
  --mode "base_model" 