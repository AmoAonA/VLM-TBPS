# VLM-TBPS

This directory is a **standalone GitHub-ready repo folder** for the
attribute-word enhanced ViPer pipeline.

Unlike the previous `open_source_release/` bundle, this one already contains the code needed to run:

- `tools/`
- `psd2/`
- `configs/`
- `scripts/`
- `data_text/`
- `data_templates/`

So if you upload **this folder itself** to GitHub, other users can clone it and run from this repo root after preparing:

- the Python environment
- the raw dataset files
- `detectron2`

Main point:

- the **open-source target here is the attribute-word branch**
- the pure `ClipViper` CUHK `0.6275` result is kept only as a **reference baseline**

## Repo structure

```text
github_open_source_repo/
├── README.md
├── ENVIRONMENT.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .flake8
├── .clang-format
├── configs/
├── scripts/
├── data_text/
├── data_templates/
├── tools/
├── psd2/
```

## Main released path

- PRW stage1 with bundled text annotations
- PRW stage2 attribute-word branch training
- PRW cache-based evaluation
- CUHK stage2 attribute-word branch training
- CUHK attribute positive-only rerank evaluation
- bundled text/schema release under `data_text/`
- dataset layout checking

## Important boundary

For the **best verified CUHK stage1 baseline**, we still recommend a **clean official ViPer checkout**.

That is why the repo also keeps:

- `scripts/run_cuhk_stage1_official.sh`

This script is only for reproducing the clean-official reference baseline separately.
It is **not** the main innovation path of this repo.

## Quick start

### 1. Prepare environment

See:

- `ENVIRONMENT.md`

### 2. Prepare dataset

Raw dataset layout expected at `${DATA_ROOT}`:

```text
${DATA_ROOT}/
├── PRW/
│   ├── query_info.txt
│   ├── frame_train.mat
│   ├── frame_test.mat
│   ├── frames/
│   └── annotations/
└── cuhk_sysu/
    ├── annotation/Images.mat
    ├── annotation/pool.mat
    ├── annotation/test/train_test/
    └── Image/SSM/
```

Bundled text / attribute assets are already shipped in this repo:

```text
github_open_source_repo/data_text/
├── PRW/
│   ├── generated_schema.json
│   ├── prw_Final_Complete_cleaned_train.json
│   └── prw_Final_Complete_cleaned_test.json
└── cuhk_sysu/
    ├── generated_schema.json
    ├── CUHK_SYSU_Final_Complete_train_cleaned.json
    └── CUHK_SYSU_Final_Complete_test_cleaned.json
```

Check:

```bash
export DATA_ROOT=/path/to/dataset
bash scripts/check_dataset_layout.sh
```

### 3. PRW stage1

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

bash scripts/run_prw_stage1.sh
```

### 4. PRW stage2

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export STAGE1_CKPT=/path/to/prw_stage1/model_final.pth

bash scripts/run_prw_stage2.sh
```

### 5. PRW cache evaluation

```bash
export PYTHON_BIN=python
export DEVICE=cuda:0
export FUSION_MODE=avg
export QUERY_BATCH_SIZE=1024
export GALLERY_FILE=/path/to/_gallery_gt_inf.pt
export QUERY_FILE=/path/to/_query_inf.pt

bash scripts/eval_prw_cache.sh
```

### 6. CUHK stage2

By default this runs stage2 training and then evaluates the best verified
CUHK attribute-word setting: global retrieval plus top/pants positive-only soft rerank.

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export STAGE1_CKPT=/path/to/official_cuhk_stage1/model_final.pth
export RUN_EVAL_AFTER=1

bash scripts/run_cuhk_stage2.sh
```

To evaluate an existing CUHK stage2 checkpoint directly:

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export CKPT=/path/to/cuhk_stage2/model_final.pth

bash scripts/eval_cuhk_stage2_posrerank.sh
```

To run the pure global diagnostic instead:

```bash
export EVAL_MODE=global
bash scripts/run_cuhk_stage2.sh
```

### 7. Smoke test

```bash
export DATA_ROOT=/path/to/dataset
bash scripts/smoke_test.sh
```

### 8. End-to-end one-iter train smoke test

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0

bash scripts/smoke_train_prw_one_iter.sh
```

This is the quickest end-to-end check that the standalone repo can really
build the model, load PRW, enter the training loop, and finish one iteration.

## Included public materials

Release-facing material now lives at repo root:

- `data_text/`
- `data_templates/`
- `configs/open_source/`
- `scripts/`

Main public configs:

- `configs/open_source/clipviper_prw_stage1_baseline.yaml`
- `configs/open_source/clipviper_cuhk_stage1_baseline.yaml`
- `configs/open_source/viper_semantic_cerberus_stage2_attr_from_stage1_bnneckfalse.yaml`
- `configs/open_source/viper_semantic_cerberus_cuhk_stage2_attr_from_stage1_bnneckfalse.yaml`
- `configs/open_source/viper_semantic_cerberus_cuhk_stage2_posrerank_eval.yaml`

## Brief result note

- `0.6275` on CUHK is the pure `ClipViper` reference baseline
- `0.6281` on CUHK is the current attribute-word positive-only rerank result
- the corresponding CUHK stage2 pure-global diagnostic is `0.6225`
- standalone result/log summary files are intentionally not bundled in this public repo

## Data note

- the JSON files under `data_text/` are dataset-derived text / attribute annotations
- raw images are **not** redistributed in this repo
- downstream use should still follow the original dataset terms for `PRW` and `CUHK-SYSU`

## Direct answer to the previous issue

If you upload only the old `open_source_release/` folder, it is **not enough**.

If you upload **this** `github_open_source_repo/` folder as a repo, then it is structurally complete enough to run the provided scripts, assuming the environment and datasets are prepared.

## Validation status

Validated with the environment described in `ENVIRONMENT.md` and dataset root supplied through `DATA_ROOT` / `PSD2_DATA_ROOT`.

Validated items:

- `bash scripts/smoke_test.sh`
- config loading for the public PRW/CUHK configs
- one-iteration PRW training smoke run via `bash scripts/smoke_train_prw_one_iter.sh`
