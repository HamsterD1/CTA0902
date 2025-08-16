## Data
The AISHELL-4 is a sizable real-recorded Mandarin speech dataset collected by 8-channel circular microphone array for speech processing in conference scenario, about 40 hours for non overlapping parts of the dataset

**Data prepare**

Use one of the following way:

- Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data_multi.sh -h
   bash local/audio2ark_multi.sh -h
   ```

Source data info will be automatically stored at `data/metainfo.json`. You can run

```bash
cd /path/to/aishell4
python utils/data/resolvedata.py
```
to refresh the information. Manually modifying is also OK.

## Result

Data prepare with command:

```bash
bash local/data_multi.sh -subsets train dev test -datapath /path/to/aishell4 
bash local/audio2ark_multi.sh train dev test --res 16000
```

Summarize experiments here.

NOTE: some of the experiments are conduct on previous code base, therefore, the settings might not be compatible to the latest. In that case, you could:

- \[Recommand\] manually modify the configuration files (`config.json` and `hyper-p.json`);

- Checkout to old code base by `hyper-p:commit` info. This could definitely reproduce the reported results, but some modules might be buggy.




### CER Results of Non-Streaming Models

The ASR architecture is identical across all models, with a size of 20.77M. All single-channel models use channel 0 as the input. EaBNet, TFGridNet, and pIGCRN perform end-to-end fine-tuning on the single-CTC, which serves as the pre-trained ASR encoder from AISHELL-4.

| Model                          | Para. of FE (M) | Aishell-4 8-ch | Aishell-4 4-ch | Aishell-4 2-ch | Alimeeting test | Alimeeting eval | XMOS test | Avg.  |
|--------------------------------|----------------|----------------|----------------|----------------|-----------------|-----------------|-----------|-------|
| single-CTC | --             | 35.65          | 35.65          | 35.65          | 40.27           | 44.10           | 87.13     | 46.41 |
| [EaBNet + CTC](exp/EaBNet+CTC)    | 3.21           | 31.79          | 35.11          | 36.97          | 35.46           | 39.32           | 93.56     | 45.37 |
| [TFGrid + CTC](exp/TFGrid+CTC)       | 0.39           | 30.51          | 33.44          | 36.61          | 33.88           | 37.18           | 85.94     | 42.93 |
| [pIGCRN + CTC](exp/pIGCRN+CTC) | 0.81           | 31.20          | 32.32          | 33.42          | 35.03           | 38.47           | 97.03     | 44.58 |
| **[SHTNet(ours)](exp/SHTNet)**               | **0.38**           | **29.34**      | **29.57**      | **31.66**      | **33.14**       | **37.01**       | **74.85** | **39.26** |


### Streaming CER (%) and Latency Results

We tested the decoding time of different models on the AISHELL-4 8-ch test set with the same chunk size (400ms) on a single 3090 GPU, recorded as Time(s). Δt(s) represents the total decoding latency introduced by the addition of the frontend. Per-utterance latency is 15.5ms.

| Model              | Time(s) | Δt(s)  | Aishell-4 8-ch | Aishell-4 4-ch | Aishell-4 2-ch | Alimeeting test | Alimeeting eval | XMOS test | Avg   |
|--------------------|---------|--------|----------------|----------------|----------------|-----------------|-----------------|-----------|-------|
| CUSIDE             | **722.83** | --     | 41.01          | 41.01          | 41.01          | 46.82           | 51.40           | 86.63     | 51.31 |
| CUSIDE-Array       | 978.26  | 255.43 | 35.08          | 37.11          | 41.54          | 39.53           | 43.08           | 82.08     | 46.40 |
| **SHTNet(ours)**   | 820.25  | **97.42** | **34.37**      | **34.74**      | **36.80**      | **37.66**       | **41.61**       | **79.50** | **44.11** |



