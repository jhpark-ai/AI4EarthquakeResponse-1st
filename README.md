# AI for Earthquake Response — 1st Place Solution

**Challenge:** [AI for Earthquake Response Challenge](https://platform.ai4eo.eu/ai-for-earthquake-response)

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Challenge Context](#challenge-context)
- [Data and Preprocessing](#data-and-preprocessing)
  - [Inputs](#inputs)
  - [Crops generation (`data_preprocess.py`)](#crops-generation-data_preprocesspy)
  - [Selection files](#selection-files)
- [Training](#training)
- [Domain Adaptation (Stress Test)](#domain-adaptation-stress-test)
- [Pipeline](#pipeline)
- [Quickstart](#quickstart)
  - [1) Preprocessing](#1-preprocessing)
  - [2) Training (5-fold)](#2-training-5-fold)
  - [3) Inference (ensemble)](#3-inference-ensemble)
- [Outputs](#outputs)
- [Dependencies](#dependencies)
- [Environment Setup (Conda)](#environment-setup-conda)
- [Notes](#notes)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository contains the official implementation of my 1st-place solution for the AI for Earthquake Response Challenge. The task is to automatically classify buildings as damaged or undamaged using pre- and post-event Very High-Resolution (VHR) satellite imagery and curated building polygons. For challenge details, see the official page: AI for Earthquake Response Challenge.

---

## Key Features

- End-to-end pipeline: preprocessing → k-fold training → ensemble inference
- Strong backbone: Eva-Large (ViT family), outperforming CNN baselines
- Robust training: extensive augmentations, BCE with positive class weighting
- Optimization: AdamW, Cosine LR schedule, mixed precision, SWA (Stochastic Weight Averaging)
- Domain adaptation: histogram matching to reduce cross-region domain shift

---

## Challenge Context

The challenge simulates rapid damage assessment after earthquakes, leveraging Earth Observation data. You are given pre/post-event imagery and building polygons for multiple scenes. Phase 1 includes fully and partially annotated scenes. Phase 2 is a stress test on previously undisclosed sites with no labels. High-level context is provided by ESA Φ-lab and the International Charter ‘Space and Major Disasters’.

---

## Data and Preprocessing

### Inputs
- Directory structure follows the original challenge bundle:
  - `Training/` or `Testing/`
    - `Annotations/All/*.gpkg` (building polygons; some with damaged labels)
    - `<Dataset>/*` containing imagery folders
  - Band folder inside each dataset: `Optical_Calibration/`
    - `r-red.tif`, `r-green.tif`, `r-blue.tif` (per-band RGB)

### Crops generation (`data_preprocess.py`)
- Generates per-building RGB crops with context padding (`PAD_PIXELS`, default 50)
- If a combined RGB exists, it is used directly; otherwise per-band TIFFs are contrast-stretched with percentile clipping (default 2%)
- A per-pixel valid-data alpha mask is embedded in the output RGB GeoTIFF
- Also writes a binary building mask GeoTIFF and a metadata CSV

### Selection files
- `config/selected_id.txt` (phase 1) and `config/selected_id_phase2.txt` (phase 2) list the datasets and annotation files used for processing

---

## Training

- Backbone: Eva-Large (ViT-based), implemented via timm
- Cross-validation: 5-fold, with ensemble at inference time
- Loss: BCEWithLogits loss with positive class weighting (computed per fold)
- Augmentations (online)
  - Mixup & CutMix, Flip, Rotate, Random Brightness, Random Gamma, Coarse Dropout, Random Mask Size
- Optimization
  - Optimizer: AdamW
  - Scheduler: Cosine annealing with warmup
  - AMP: Mixed precision training
  - SWA: Stochastic Weight Averaging for better generalization

---

## Domain Adaptation (Stress Test)

We mitigate domain shift by applying histogram matching (see `hist_matching.png`) to align color distributions between the training and test scenes. In practice, we match per-channel histograms from the training distribution to the target (test) distribution using a small subset of test tiles as references. This preserves spatial structure while normalizing tonal differences across satellites/regions.

![Histogram Matching for Domain Adaptation](hist_matching.png)

---

## Pipeline

1) Preprocess crops
   - Set `PHASE` and configs in `data_preprocess.py`
   - Example outputs: `CROPS_50_all` or `CROPS_50_all_phase2`
2) Train with 5-fold CV
   - Produces 5 checkpoints under `saved_model/`
3) Inference (ensemble of 5 folds)
   - Writes predictions back to `.gpkg` with a `damaged` column

---

## Quickstart

### 1) Preprocessing

- Configure phase and selection file paths in `data_preprocess.py`:
  - `PHASE=1` → `DATA_ROOT=Training`, `SELECT_FILE=config/selected_id.txt`, `OUT_ROOT=CROPS_50_all`
  - `PHASE=2` → `DATA_ROOT=Testing`, `SELECT_FILE=config/selected_id_phase2.txt`, `OUT_ROOT=CROPS_50_all_phase2`
- Run:

```bash
python3 data_preprocess.py
