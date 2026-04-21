# Environment

## Minimum practical environment

This repo expects:

- Python `3.10`
- PyTorch with CUDA support
- `torchvision`
- `detectron2`
- `fvcore`
- `iopath`
- `yacs`
- `opencv-python`
- `scipy`
- `Pillow`
- `tqdm`
- `pyyaml`

## Verified environment from the clean CUHK stage1 run

The clean official CUHK baseline was evaluated in an environment containing:

- Python `3.10.20`
- PyTorch `2.4.0+cu121`
- torchvision `0.19.0+cu121`
- detectron2 `0.6`
- CUDA runtime `12.1`

## Detectron2 note

This repo imports `detectron2` as an installed Python package.

So cloning this repo is **not** enough by itself; you still need a working `detectron2` installation compatible with your PyTorch/CUDA version.

## Recommended workflow

1. Create a fresh Python 3.10 environment.
2. Install matching `torch` / `torchvision`.
3. Install a matching `detectron2`.
4. Install the remaining Python dependencies.
5. Run:

```bash
python tools/train_ps_net.py --help
python tools/eval_search_from_cache.py --help
```

If both commands work, the code entrypoints are installed correctly.

## Dataset note

The semantic branch relies on both:

- cleaned text annotations
- generated schema files

So you must prepare the JSON/TXT files documented in:

- `README.md`
- `data_templates/`
