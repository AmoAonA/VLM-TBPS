# Closing the Modality Gap with Symbols: Explicit Attribute Alignment for End-to-End Text-Based Person Search

**Official implementation for ACM MM 2026**

**中文标题：以符号弥合模态鸿沟：面向端到端文本行人搜索的显式属性对齐**

[[English]](#english) | [[中文]](#中文)

---

<a name="english"></a>
## English

Official implementation of **"Closing the Modality Gap with Symbols: Explicit Attribute Alignment for End-to-End Text-Based Person Search"**, accepted to **ACM MM 2026**.

### Overview

This repository contains our attribute-word enhanced text-based person search framework built on top of the ViPer pipeline. The released code supports:

- CUHK-SYSU stage1 baseline
- PRW stage1 training
- PRW stage2 attribute-word branch training
- PRW full-checkpoint evaluation
- CUHK-SYSU stage2 attribute-word branch training
- CUHK-SYSU positive-only rerank evaluation
- bundled text / schema files under `data_text/`

### Requirements

- Python >= 3.8
- PyTorch
- torchvision
- `detectron2`

Environment details are summarized in `ENVIRONMENT.md`.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Data Preparation

Prepare the raw datasets under `${DATA_ROOT}`:

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

This repository already includes the processed text / attribute files:

```text
data_text/
├── PRW/
│   ├── generated_schema.json
│   ├── prw_Final_Complete_cleaned_train.json
│   └── prw_Final_Complete_cleaned_test.json
└── cuhk_sysu/
    ├── generated_schema.json
    ├── CUHK_SYSU_Final_Complete_train_cleaned.json
    └── CUHK_SYSU_Final_Complete_test_cleaned.json
```

Check dataset layout:

```bash
export DATA_ROOT=/path/to/dataset
bash scripts/check_dataset_layout.sh
```

### Training

#### CUHK-SYSU stage1 baseline

This is the stage1 baseline used before our stage2 attribute-word branch
training. The released config is aligned with the official ViPer
`clipviper_cuhk.yaml` setting.

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export RUN_EVAL_AFTER=1

bash scripts/run_cuhk_stage1.sh
```

#### PRW stage1

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

bash scripts/run_prw_stage1.sh
```

#### PRW stage2

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export STAGE1_CKPT=/path/to/prw_stage1/model_final.pth
export RUN_EVAL_AFTER=1

bash scripts/run_prw_stage2.sh
```

#### CUHK-SYSU stage2

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export STAGE1_CKPT=/path/to/cuhk_stage1/model_final.pth
export RUN_EVAL_AFTER=1

bash scripts/run_cuhk_stage2.sh
```

### Evaluation

#### PRW stage2 checkpoint evaluation

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export CKPT=/path/to/prw_stage2/model_final.pth
export FUSION_MODE=avg

bash scripts/eval_prw_stage2.sh
```

#### CUHK-SYSU stage2 checkpoint evaluation

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export CKPT=/path/to/cuhk_stage2/model_final.pth

bash scripts/eval_cuhk_stage2_posrerank.sh
```

To run the pure global diagnostic:

```bash
export EVAL_MODE=global
bash scripts/run_cuhk_stage2.sh
```

### Project Structure

```text
├── configs/            # Configuration files
├── data_templates/     # Public data format examples
├── data_text/          # Released text / attribute annotations
├── psd2/               # Core library
├── scripts/            # Training / evaluation shell scripts
├── tools/              # Main Python entrypoint
├── ENVIRONMENT.md      # Environment notes
├── requirements.txt    # Python dependencies
└── README.md
```

### Acknowledgments

This codebase is built on top of the ViPer / Detectron-style person search framework. We thank the original authors and open-source contributors for their valuable work.

---

<a name="中文"></a>
## 中文

这是论文 **《以符号弥合模态鸿沟：面向端到端文本行人搜索的显式属性对齐》** 的官方实现，发表于 **ACM MM 2026**。

### 概述

本仓库实现了基于 ViPer 管线扩展的属性词增强文本行人搜索框架，当前公开内容包括：

- CUHK-SYSU 一阶段 baseline
- PRW 一阶段训练
- PRW 二阶段属性词分支训练
- PRW 整 ckpt 评测
- CUHK-SYSU 二阶段属性词分支训练
- CUHK-SYSU 正向 rerank 评测
- `data_text/` 中附带的文本 / 属性 / schema 文件

### 环境要求

- Python >= 3.8
- PyTorch
- torchvision
- `detectron2`

环境说明见 `ENVIRONMENT.md`。

安装依赖：

```bash
pip install -r requirements.txt
```

### 数据准备

将原始数据集放在 `${DATA_ROOT}` 下：

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

仓库中已经附带整理好的文本 / 属性文件：

```text
data_text/
├── PRW/
│   ├── generated_schema.json
│   ├── prw_Final_Complete_cleaned_train.json
│   └── prw_Final_Complete_cleaned_test.json
└── cuhk_sysu/
    ├── generated_schema.json
    ├── CUHK_SYSU_Final_Complete_train_cleaned.json
    └── CUHK_SYSU_Final_Complete_test_cleaned.json
```

检查数据目录：

```bash
export DATA_ROOT=/path/to/dataset
bash scripts/check_dataset_layout.sh
```

### 训练

#### CUHK-SYSU 一阶段 baseline

这是后续二阶段属性词分支训练所使用的一阶段基础模型。公开配置与官方 ViPer 的
`clipviper_cuhk.yaml` 保持对齐。

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export RUN_EVAL_AFTER=1

bash scripts/run_cuhk_stage1.sh
```

#### PRW 一阶段

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

bash scripts/run_prw_stage1.sh
```

#### PRW 二阶段

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export STAGE1_CKPT=/path/to/prw_stage1/model_final.pth
export RUN_EVAL_AFTER=1

bash scripts/run_prw_stage2.sh
```

#### CUHK-SYSU 二阶段

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export STAGE1_CKPT=/path/to/cuhk_stage1/model_final.pth
export RUN_EVAL_AFTER=1

bash scripts/run_cuhk_stage2.sh
```

### 评测

#### PRW 二阶段整 ckpt 评测

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export CKPT=/path/to/prw_stage2/model_final.pth
export FUSION_MODE=avg

bash scripts/eval_prw_stage2.sh
```

#### CUHK-SYSU 二阶段模型评测

```bash
export DATA_ROOT=/path/to/dataset
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0
export CKPT=/path/to/cuhk_stage2/model_final.pth

bash scripts/eval_cuhk_stage2_posrerank.sh
```

如果要跑纯 global 诊断：

```bash
export EVAL_MODE=global
bash scripts/run_cuhk_stage2.sh
```

### 项目结构

```text
├── configs/            # 配置文件
├── data_templates/     # 数据格式示例
├── data_text/          # 开源的文本 / 属性标注
├── psd2/               # 核心代码库
├── scripts/            # 训练 / 评测脚本
├── tools/              # 主 Python 入口
├── ENVIRONMENT.md      # 环境说明
├── requirements.txt    # Python 依赖
└── README.md
```

### 致谢

本仓库基于 ViPer / Detectron 风格的行人搜索框架扩展实现，感谢相关开源工作与原作者的贡献。
