# LLMResearch — RunPod Setup Guide

This repo is a fork of [HuggingFace Transformers](https://github.com/huggingface/transformers) with custom attention amplification research on LLaMA models, plus benchmark and inference scripts.

## Quickstart on a New RunPod Instance

```bash
git clone https://github.com/manoharvellala/LLMResearch.git /workspace/transformers
export HF_TOKEN=your_token
bash /workspace/transformers/research/setup.sh
```

That's it — environment is ready.

## Notes

- The `/workspace` directory itself is **not** a git repo. Only `/workspace/transformers` is. Running `git status` in `/workspace` will give a fatal error — this is expected and fine.
- Always `cd /workspace/transformers` before running any git commands.

## Research Files

| File | Description |
|------|-------------|
| `research/benchmark.py` | Privacy eval benchmark against LLaMA |
| `research/infer.py` | Inference helper |
| `research/setup.sh` | Restores Python environment after RunPod resume |

## Key Changes

- `src/transformers/models/llama/modeling_llama.py` — attention amplification modifications
