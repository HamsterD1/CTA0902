### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 21.58 M
* GPU info \[9\]
  * \[9\] NVIDIA GeForce RTX 3090

### Notes

* 

### Result
```
best-5
CH8
test_raw        %SER 88.85 | %CER 31.20 [ 40962 / 131298, 4549 ins, 4935 del, 31478 sub ]
  streaming
  test_raw        %SER 97.53 | %CER 59.52 [ 78144 / 131298, 896 ins, 42979 del, 34269 sub ]

CH4
test_raw        %SER 89.49 | %CER 32.32 [ 42434 / 131298, 4772 ins, 4952 del, 32710 sub ]
  streaming
  test_raw        %SER 97.56 | %CER 60.78 [ 79808 / 131298, 866 ins, 44379 del, 34563 sub ]

CH2
test_raw        %SER 89.81 | %CER 33.42 [ 43878 / 131298, 4824 ins, 5374 del, 33680 sub ]
  streaming
  test_raw        %SER 97.53 | %CER 62.00 [ 81406 / 131298, 822 ins, 46147 del, 34437 sub ]


```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

Non-streaming
T:1
Model FLOPs: 12.21 GFLOPs
Model Parameters: 0.87 M
T:2
Model FLOPs: 24.42 GFLOPs
Model Parameters: 0.87 M
T:3
Model FLOPs: 36.63 GFLOPs
Model Parameters: 0.87 M
T:4
Model FLOPs: 48.84 GFLOPs
Model Parameters: 0.87 M
T:5
Model FLOPs: 61.05 GFLOPs
Model Parameters: 0.87 M
T:6
Model FLOPs: 73.25 GFLOPs
Model Parameters: 0.87 M
T:7
Model FLOPs: 85.46 GFLOPs
Model Parameters: 0.87 M
T:8
Model FLOPs: 97.67 GFLOPs
Model Parameters: 0.87 M
T:9
Model FLOPs: 109.88 GFLOPs
Model Parameters: 0.87 M
T:10
Model FLOPs: 122.09 GFLOPs
Model Parameters: 0.87 M

IGCRN(
  873.89 K = 100% Params, 72.91 GMACs = 100% MACs, 290.27 GFLOPS = 100% FLOPs
  (act): ELU(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs, alpha=1.0)
  (e1_sph): convGLU(
    35.72 K = 4.09% Params, 3.08 GMACs = 4.23% MACs, 6.25 GFLOPS = 2.15% FLOPs
    (conv): Conv2d(5 K = 0.57% Params, 1.54 GMACs = 2.12% MACs, 3.08 GFLOPS = 1.06% FLOPs, 50, 20, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), bias=False, padding_mode=circular)
    (convGate): Conv2d(5.02 K = 0.57% Params, 1.54 GMACs = 2.12% MACs, 3.09 GFLOPS = 1.06% FLOPs, 50, 20, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      25.7 K = 2.94% Params, 0 MACs = 0% MACs, 77.1 MFLOPS = 0.03% FLOPs
      (ln): LayerNorm(25.7 K = 2.94% Params, 0 MACs = 0% MACs, 77.1 MFLOPS = 0.03% FLOPs, (50, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e2_sph): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e3_sph): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(8, 0), dilation=(4, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(8, 0), dilation=(4, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e4_sph): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(16, 0), dilation=(8, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(16, 0), dilation=(8, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e5_sph): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(32, 0), dilation=(16, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(32, 0), dilation=(16, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e6_sph): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(64, 0), dilation=(32, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(64, 0), dilation=(32, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e1_stft): convGLU(
    11.44 K = 1.31% Params, 986.88 MMACs = 1.35% MACs, 2 GFLOPS = 0.69% FLOPs
    (conv): Conv2d(1.6 K = 0.18% Params, 493.44 MMACs = 0.68% MACs, 986.88 MFLOPS = 0.34% FLOPs, 16, 20, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), bias=False, padding_mode=circular)
    (convGate): Conv2d(1.62 K = 0.19% Params, 493.44 MMACs = 0.68% MACs, 993.05 MFLOPS = 0.34% FLOPs, 16, 20, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      8.22 K = 0.94% Params, 0 MACs = 0% MACs, 24.67 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(8.22 K = 0.94% Params, 0 MACs = 0% MACs, 24.67 MFLOPS = 0.01% FLOPs, (16, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e2_stft): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e3_stft): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(8, 0), dilation=(4, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(8, 0), dilation=(4, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e4_stft): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(16, 0), dilation=(8, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(16, 0), dilation=(8, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e5_stft): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(32, 0), dilation=(16, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(32, 0), dilation=(16, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (e6_stft): convGLU(
    14.3 K = 1.64% Params, 1.23 GMACs = 1.69% MACs, 2.5 GFLOPS = 0.86% FLOPs
    (conv): Conv2d(2 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.23 GFLOPS = 0.42% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(64, 0), dilation=(32, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(2.02 K = 0.23% Params, 616.8 MMACs = 0.85% MACs, 1.24 GFLOPS = 0.43% FLOPs, 20, 20, kernel_size=(5, 1), stride=(1, 1), padding=(64, 0), dilation=(32, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs
      (ln): LayerNorm(10.28 K = 1.18% Params, 0 MACs = 0% MACs, 30.84 MFLOPS = 0.01% FLOPs, (20, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (BNe6): LayerNorm(
    20.56 K = 2.35% Params, 0 MACs = 0% MACs, 61.68 MFLOPS = 0.02% FLOPs
    (ln): LayerNorm(20.56 K = 2.35% Params, 0 MACs = 0% MACs, 61.68 MFLOPS = 0.02% FLOPs, (40, 257), eps=1e-05, elementwise_affine=True)
  )
  (ch_lstm): ch_lstm(
    239.4 K = 27.39% Params, 1.97 GMACs = 2.71% MACs, 147.05 GFLOPS = 50.66% FLOPs
    (lstm2): LSTM(232.96 K = 26.66% Params, 0 MACs = 0% MACs, 143.1 GFLOPS = 49.3% FLOPs, 40, 80, num_layers=2, batch_first=True, bidirectional=True)
    (linear_lstm_out2): Linear(6.44 K = 0.74% Params, 1.97 GMACs = 2.71% MACs, 3.95 GFLOPS = 1.36% FLOPs, in_features=160, out_features=40, bias=True)
  )
  (d16): convGLU(
    36.6 K = 4.19% Params, 4.93 GMACs = 6.77% MACs, 9.94 GFLOPS = 3.43% FLOPs
    (conv): Conv2d(8 K = 0.92% Params, 2.47 GMACs = 3.38% MACs, 4.93 GFLOPS = 1.7% FLOPs, 40, 40, kernel_size=(5, 1), stride=(1, 1), padding=(64, 0), dilation=(32, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(8.04 K = 0.92% Params, 2.47 GMACs = 3.38% MACs, 4.95 GFLOPS = 1.7% FLOPs, 40, 40, kernel_size=(5, 1), stride=(1, 1), padding=(64, 0), dilation=(32, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      20.56 K = 2.35% Params, 0 MACs = 0% MACs, 61.68 MFLOPS = 0.02% FLOPs
      (ln): LayerNorm(20.56 K = 2.35% Params, 0 MACs = 0% MACs, 61.68 MFLOPS = 0.02% FLOPs, (40, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (d15): convGLU(
    73.16 K = 8.37% Params, 9.87 GMACs = 13.54% MACs, 19.87 GFLOPS = 6.85% FLOPs
    (conv): Conv2d(16 K = 1.83% Params, 4.93 GMACs = 6.77% MACs, 9.87 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(32, 0), dilation=(16, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(16.04 K = 1.84% Params, 4.93 GMACs = 6.77% MACs, 9.88 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(32, 0), dilation=(16, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs
      (ln): LayerNorm(41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs, (80, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (d14): convGLU(
    73.16 K = 8.37% Params, 9.87 GMACs = 13.54% MACs, 19.87 GFLOPS = 6.85% FLOPs
    (conv): Conv2d(16 K = 1.83% Params, 4.93 GMACs = 6.77% MACs, 9.87 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(16, 0), dilation=(8, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(16.04 K = 1.84% Params, 4.93 GMACs = 6.77% MACs, 9.88 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(16, 0), dilation=(8, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs
      (ln): LayerNorm(41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs, (80, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (d13): convGLU(
    73.16 K = 8.37% Params, 9.87 GMACs = 13.54% MACs, 19.87 GFLOPS = 6.85% FLOPs
    (conv): Conv2d(16 K = 1.83% Params, 4.93 GMACs = 6.77% MACs, 9.87 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(8, 0), dilation=(4, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(16.04 K = 1.84% Params, 4.93 GMACs = 6.77% MACs, 9.88 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(8, 0), dilation=(4, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs
      (ln): LayerNorm(41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs, (80, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (d12): convGLU(
    73.16 K = 8.37% Params, 9.87 GMACs = 13.54% MACs, 19.87 GFLOPS = 6.85% FLOPs
    (conv): Conv2d(16 K = 1.83% Params, 4.93 GMACs = 6.77% MACs, 9.87 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1), bias=False, padding_mode=circular)
    (convGate): Conv2d(16.04 K = 1.84% Params, 4.93 GMACs = 6.77% MACs, 9.88 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), dilation=(2, 1), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs
      (ln): LayerNorm(41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs, (80, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (d11): convGLU(
    73.16 K = 8.37% Params, 9.87 GMACs = 13.54% MACs, 19.87 GFLOPS = 6.85% FLOPs
    (conv): Conv2d(16 K = 1.83% Params, 4.93 GMACs = 6.77% MACs, 9.87 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), bias=False, padding_mode=circular)
    (convGate): Conv2d(16.04 K = 1.84% Params, 4.93 GMACs = 6.77% MACs, 9.88 GFLOPS = 3.4% FLOPs, 80, 40, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs
      (ln): LayerNorm(41.12 K = 4.71% Params, 0 MACs = 0% MACs, 123.36 MFLOPS = 0.04% FLOPs, (80, 257), eps=1e-05, elementwise_affine=True)
    )
  )
  (convOUT): convGLU(
    21.36 K = 2.44% Params, 246.72 MMACs = 0.34% MACs, 555.74 MFLOPS = 0.19% FLOPs
    (conv): Conv2d(400 = 0.05% Params, 123.36 MMACs = 0.17% MACs, 246.72 MFLOPS = 0.08% FLOPs, 40, 2, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), bias=False, padding_mode=circular)
    (convGate): Conv2d(402 = 0.05% Params, 123.36 MMACs = 0.17% MACs, 247.34 MFLOPS = 0.09% FLOPs, 40, 2, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), padding_mode=circular)
    (gate_act): Sigmoid(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
    (LN): LayerNorm(
      20.56 K = 2.35% Params, 0 MACs = 0% MACs, 61.68 MFLOPS = 0.02% FLOPs
      (ln): LayerNorm(20.56 K = 2.35% Params, 0 MACs = 0% MACs, 61.68 MFLOPS = 0.02% FLOPs, (40, 257), eps=1e-05, elementwise_affine=True)
    )
  )
)
---------------------------------------------------------------------------------------------------

主要是LSTM与D11~D16