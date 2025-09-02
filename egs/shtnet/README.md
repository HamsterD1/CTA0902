# 1. Overview

This directory contains the implementation code for the INTERSPEECH 2025 paper "Lightweight and Robust Multi-Channel End-to-End Speech Recognition with Spherical Harmonic Transform". The code reproduces the experiments on the AISHELL-4 dataset, covering data preprocessing, model training, and evaluation steps for multi-channel ASR systems.

# 2. Results

## 2.1 Non-Streaming CER Results

The ASR architecture is identical across all models, with a size of 20.77M. All single-channel models use channel 0 as the input. EaBNet, TFGridNet, and pIGCRN perform end-to-end fine-tuning on the single-CTC, which serves as the pre-trained ASR encoder from AISHELL-4.

| Model                          | Para. of FE (M) | Aishell-4 8-ch | Aishell-4 4-ch | Aishell-4 2-ch | Alimeeting test | Alimeeting eval | XMOS test | Avg.  |
|--------------------------------|----------------|----------------|----------------|----------------|-----------------|-----------------|-----------|-------|
| single-CTC | --             | 35.65          | 35.65          | 35.65          | 40.27           | 44.10           | 87.13     | 46.41 |
| [EaBNet + CTC](exp/EaBNet+CTC)    | 3.21           | 31.79          | 35.11          | 36.97          | 35.46           | 39.32           | 93.56     | 45.37 |
| [TFGrid + CTC](exp/TFGrid+CTC)       | 0.39           | 30.51          | 33.44          | 36.61          | 33.88           | 37.18           | 85.94     | 42.93 |
| [pIGCRN + CTC](exp/pIGCRN+CTC) | 0.81           | 31.20          | 32.32          | 33.42          | 35.03           | 38.47           | 97.03     | 44.58 |
| **[SHTNet(ours)](exp/SHTNet)**               | **0.38**           | **29.34**      | **29.57**      | **31.66**      | **33.14**       | **37.01**       | **74.85** | **39.26** |

## 2.2 Streaming CER and Latency Results

We tested the decoding time of different models on the AISHELL-4 8-ch test set with the same chunk size (400ms) on a single 3090 GPU, recorded as Time(s). 
**Key latency metrics**:
- **Δt(s)**: Total frontend decoding latency for entire test set
- **Per-utterance latency**: Average additional latency per utterance = 15.5ms

| Model              | Time(s) | Δt(s)  | Aishell-4 8-ch | Aishell-4 4-ch | Aishell-4 2-ch | Alimeeting test | Alimeeting eval | XMOS test | Avg   |
|--------------------|---------|--------|----------------|----------------|----------------|-----------------|-----------------|-----------|-------|
| CUSIDE             | **722.83** | --     | 41.01          | 41.01          | 41.01          | 46.82           | 51.40           | 86.63     | 51.31 |
| CUSIDE-Array       | 978.26  | 255.43 | 35.08          | 37.11          | 41.54          | 39.53           | 43.08           | 82.08     | 46.40 |
| **SHTNet(ours)**   | 820.25  | **97.42** | **34.37**      | **34.74**      | **36.80**      | **37.66**       | **41.61**       | **79.50** | **44.11** |

# 3. Code Structure

Key directories:
- **`exp/`**: Model implementations
  - `EaBNet+CTC`: EaBNet with single-CTC backend
  - `pIGCRN+CTC`: parall-IGCRN with single-CTC backend
  - `SHTNet`: Proposed SHTNet implementation
  - `TFGrid+CTC`: TFGridNet with single-CTC backend
- **`data`**: contains meta data, speech features, LM and FST graph files.
- **`local`**: contains some trial scripts.
- **`cat` and `utils`**: include codes for ASR training and evaluation in CAT.

# 4. How to Use

1. **Data Preparation**
   Follow Section 3 instructions to process AISHELL-4 data
2. **Model Training**
    - Refer to the instructions provided in the experiment directory (the link is available in the result table).
    - Proceed to train your own model accordingly.
3. **Evaluation without Language Model (LM)**
    - To evaluate the Phoneme Error Rate (PER) and Word Error Rate (WER) without an LM, follow the instructions given in the experiment directory (linked in the result table).

For detailed instructions, refer to:
[Lightweight and Robust Multi-Channel End-to-End Speech Recognition with Spherical Harmonic Transform](https://arxiv.org/abs/2506.11630)

