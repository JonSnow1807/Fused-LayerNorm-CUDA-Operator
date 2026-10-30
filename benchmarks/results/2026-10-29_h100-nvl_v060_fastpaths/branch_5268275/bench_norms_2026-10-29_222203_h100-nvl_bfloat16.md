# bench_norms (2026-10-29T22:20:19+00:00)

NVIDIA H100 NVL, torch 2.13.0+cu130, CUDA 13.0, commit 526827532758edcb11ea7315208b3a5a214ab217 (dirty=True), driver 580.173.02.

| op | MxN | dtype | candidate | eager us | kernel us | GB/s | % peak |
|---|---|---|---|---:|---:|---:|---:|
| rms_norm | 32x768 | bfloat16 | fused_layernorm.rms_norm | 9.05 | 1.88 | 53.0 | n/a |
| rms_norm | 32x768 | bfloat16 | F.rms_norm (aten fused kernel) | 7.26 | 2.18 | 45.8 | n/a |
| rms_norm | 32x768 | bfloat16 | torch.compile(F.rms_norm) | 31.35 | 1.19 | 84.1 | n/a |
| rms_norm | 32x1024 | bfloat16 | fused_layernorm.rms_norm | 11.68 | 1.88 | 70.6 | n/a |
| rms_norm | 32x1024 | bfloat16 | F.rms_norm (aten fused kernel) | 10.69 | 2.19 | 60.8 | n/a |
| rms_norm | 32x1024 | bfloat16 | torch.compile(F.rms_norm) | 34.57 | 1.19 | 111.9 | n/a |
| rms_norm | 32x4096 | bfloat16 | fused_layernorm.rms_norm | 11.79 | 2.29 | 232.6 | n/a |
| rms_norm | 32x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.77 | 4.59 | 115.9 | n/a |
| rms_norm | 32x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.33 | 1.61 | 330.5 | n/a |
| rms_norm | 17x1023 | bfloat16 | fused_layernorm.rms_norm | 11.69 | 1.87 | 38.2 | n/a |
| rms_norm | 17x1023 | bfloat16 | F.rms_norm (aten fused kernel) | 13.12 | 4.76 | 15.1 | n/a |
| rms_norm | 17x1023 | bfloat16 | torch.compile(F.rms_norm) | 34.86 | 1.29 | 55.5 | n/a |
| rms_norm | 128x4096 | bfloat16 | fused_layernorm.rms_norm | 11.44 | 2.61 | 807.6 | n/a |
| rms_norm | 128x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.56 | 4.91 | 429.0 | n/a |
| rms_norm | 128x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.12 | 2.14 | 982.6 | n/a |
| rms_norm | 512x1024 | bfloat16 | fused_layernorm.rms_norm | 11.53 | 2.75 | 763.1 | n/a |
| rms_norm | 512x1024 | bfloat16 | F.rms_norm (aten fused kernel) | 10.59 | 2.82 | 743.5 | n/a |
| rms_norm | 512x1024 | bfloat16 | torch.compile(F.rms_norm) | 34.48 | 1.75 | 1200.2 | n/a |
| rms_norm | 1x4096 | bfloat16 | fused_layernorm.rms_norm | 11.60 | 2.15 | 11.4 | n/a |
| rms_norm | 1x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.68 | 4.42 | 5.6 | n/a |
| rms_norm | 1x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.65 | 1.46 | 16.8 | n/a |
| rms_norm | 16x4096 | bfloat16 | fused_layernorm.rms_norm | 11.63 | 2.23 | 121.2 | n/a |
| rms_norm | 16x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.74 | 4.52 | 59.8 | n/a |
| rms_norm | 16x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.16 | 1.77 | 152.4 | n/a |
| rms_norm | 64x4096 | bfloat16 | fused_layernorm.rms_norm | 11.51 | 2.37 | 445.0 | n/a |
| rms_norm | 64x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.63 | 4.72 | 223.9 | n/a |
| rms_norm | 64x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.57 | 1.69 | 624.4 | n/a |
| rms_norm | 192x4096 | bfloat16 | fused_layernorm.rms_norm | 11.62 | 3.14 | 1005.8 | n/a |
| rms_norm | 192x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.73 | 5.04 | 626.0 | n/a |
| rms_norm | 192x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.85 | 2.39 | 1317.9 | n/a |
| rms_norm | 200x4096 | bfloat16 | fused_layernorm.rms_norm | 11.54 | 3.16 | 1039.3 | n/a |
| rms_norm | 200x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.68 | 5.06 | 649.2 | n/a |
| rms_norm | 200x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.28 | 2.15 | 1527.0 | n/a |
| rms_norm | 255x4096 | bfloat16 | fused_layernorm.rms_norm | 11.56 | 3.26 | 1285.2 | n/a |
| rms_norm | 255x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.71 | 5.12 | 816.8 | n/a |
| rms_norm | 255x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.43 | 2.25 | 1857.6 | n/a |
| rms_norm | 256x4096 | bfloat16 | fused_layernorm.rms_norm | 11.62 | 2.98 | 1412.5 | n/a |
| rms_norm | 256x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.67 | 5.16 | 814.4 | n/a |
| rms_norm | 256x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.40 | 2.25 | 1863.9 | n/a |
| rms_norm | 257x4096 | bfloat16 | fused_layernorm.rms_norm | 11.57 | 3.02 | 1396.3 | n/a |
| rms_norm | 257x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.72 | 5.15 | 819.8 | n/a |
| rms_norm | 257x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.38 | 2.59 | 1631.1 | n/a |
| rms_norm | 64x8192 | bfloat16 | fused_layernorm.rms_norm | 11.57 | 2.82 | 748.3 | n/a |
| rms_norm | 64x8192 | bfloat16 | F.rms_norm (aten fused kernel) | 10.55 | 7.93 | 266.4 | n/a |
| rms_norm | 64x8192 | bfloat16 | torch.compile(F.rms_norm) | 35.19 | 2.16 | 979.9 | n/a |
| rms_norm | 32x16384 | bfloat16 | fused_layernorm.rms_norm | 11.55 | 3.62 | 588.7 | n/a |
| rms_norm | 32x16384 | bfloat16 | F.rms_norm (aten fused kernel) | 15.73 | 14.44 | 147.6 | n/a |
| rms_norm | 32x16384 | bfloat16 | torch.compile(F.rms_norm) | 47.90 | 4.33 | 491.6 | n/a |
| rms_norm | 4096x4096 | bfloat16 | fused_layernorm.rms_norm | 23.28 | 21.56 | 3113.5 | n/a |
| rms_norm | 4096x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 25.91 | 24.65 | 2723.3 | n/a |
| rms_norm | 4096x4096 | bfloat16 | torch.compile(F.rms_norm) | 35.28 | 24.23 | 2769.4 | n/a |
| rms_norm | 8192x1024 | bfloat16 | fused_layernorm.rms_norm | 11.73 | 8.54 | 3928.1 | n/a |
| rms_norm | 8192x1024 | bfloat16 | F.rms_norm (aten fused kernel) | 12.52 | 11.57 | 2901.3 | n/a |
| rms_norm | 8192x1024 | bfloat16 | torch.compile(F.rms_norm) | 35.32 | 8.81 | 3807.6 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | fused op | 14.18 | 1.91 | 103.9 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | fused op (inplace) | 12.41 | 1.91 | 104.0 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | eager composite (add; norm) | 17.55 | 3.30 | 60.0 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | torch.compile(composite) | 38.22 | 1.23 | 161.7 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | fused op | 14.10 | 1.92 | 137.3 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | fused op (inplace) | 12.31 | 1.92 | 137.5 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | eager composite (add; norm) | 17.45 | 3.25 | 81.2 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | torch.compile(composite) | 38.60 | 1.24 | 212.4 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | fused op | 14.24 | 2.28 | 463.6 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | fused op (inplace) | 12.41 | 2.27 | 464.8 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | eager composite (add; norm) | 17.59 | 6.09 | 173.5 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | torch.compile(composite) | 38.44 | 1.80 | 586.4 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | fused op | 14.17 | 2.02 | 69.9 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | fused op (inplace) | 12.27 | 2.00 | 70.6 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | eager composite (add; norm) | 20.06 | 6.23 | 22.6 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | torch.compile(composite) | 38.81 | 1.44 | 97.8 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | fused op | 14.28 | 2.65 | 1584.3 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | fused op (inplace) | 12.54 | 2.66 | 1582.4 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | eager composite (add; norm) | 17.63 | 6.86 | 612.6 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | torch.compile(composite) | 38.56 | 2.48 | 1696.4 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | fused op | 14.09 | 3.00 | 1397.7 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | fused op (inplace) | 12.35 | 3.01 | 1395.7 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | eager composite (add; norm) | 17.41 | 4.39 | 955.1 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | torch.compile(composite) | 38.02 | 1.93 | 2172.6 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | fused op | 14.09 | 2.07 | 19.7 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | fused op (inplace) | 12.39 | 2.07 | 19.8 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | eager composite (add; norm) | 17.57 | 5.36 | 7.6 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | torch.compile(composite) | 38.16 | 1.64 | 25.0 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | fused op | 14.35 | 2.20 | 242.5 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | fused op (inplace) | 12.54 | 2.19 | 243.0 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | eager composite (add; norm) | 17.81 | 5.99 | 89.0 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | torch.compile(composite) | 38.61 | 1.74 | 306.5 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | fused op | 14.20 | 2.39 | 881.9 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | fused op (inplace) | 12.38 | 2.38 | 883.0 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | eager composite (add; norm) | 17.61 | 6.37 | 330.7 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | torch.compile(composite) | 38.37 | 2.19 | 959.3 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | fused op | 14.10 | 3.15 | 2000.3 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | fused op (inplace) | 12.34 | 3.18 | 1983.8 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | eager composite (add; norm) | 17.47 | 7.31 | 861.8 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | torch.compile(composite) | 37.81 | 2.75 | 2286.9 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | fused op | 14.11 | 3.17 | 2068.8 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | fused op (inplace) | 12.37 | 3.18 | 2064.8 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | eager composite (add; norm) | 17.45 | 7.33 | 894.6 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | torch.compile(composite) | 38.14 | 2.56 | 2568.1 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | fused op | 14.11 | 3.29 | 2539.4 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | fused op (inplace) | 12.30 | 3.27 | 2556.0 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | eager composite (add; norm) | 17.53 | 7.59 | 1102.3 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | torch.compile(composite) | 37.90 | 2.66 | 3139.0 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | fused op | 14.24 | 3.52 | 2384.0 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | fused op (inplace) | 12.47 | 3.47 | 2417.0 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | eager composite (add; norm) | 17.58 | 7.70 | 1090.2 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | torch.compile(composite) | 38.06 | 2.68 | 3127.6 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | fused op | 14.31 | 3.54 | 2382.6 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | fused op (inplace) | 12.45 | 3.52 | 2394.0 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | eager composite (add; norm) | 17.69 | 7.65 | 1102.6 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | torch.compile(composite) | 37.97 | 2.63 | 3199.3 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | fused op | 14.10 | 3.23 | 1304.0 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | fused op (inplace) | 12.45 | 3.26 | 1291.5 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | eager composite (add; norm) | 17.47 | 10.40 | 404.9 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | torch.compile(composite) | 37.99 | 2.40 | 1752.5 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | fused op | 14.03 | 4.70 | 899.3 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | fused op (inplace) | 12.29 | 4.74 | 891.6 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | eager composite (add; norm) | 19.96 | 17.55 | 240.9 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | torch.compile(composite) | 50.73 | 4.92 | 859.9 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | fused op | 43.16 | 41.48 | 3235.6 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | fused op (inplace) | 42.85 | 45.17 | 2971.8 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | eager composite (add; norm) | 59.15 | 62.42 | 2150.3 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | torch.compile(composite) | 44.68 | 41.85 | 3207.2 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | fused op | 22.90 | 20.87 | 3216.3 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | fused op (inplace) | 20.17 | 18.85 | 3560.4 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | eager composite (add; norm) | 30.74 | 28.42 | 2361.1 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | torch.compile(composite) | 38.98 | 19.05 | 3523.6 | n/a |
| rms_norm_fp8_dynamic | 32x768 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.29 | 2.49 | 30.3 | n/a |
| rms_norm_fp8_dynamic | 32x768 | bfloat16 | eager composite (norm; amax; quant) | 98.06 | 18.96 | 4.0 | n/a |
| rms_norm_fp8_dynamic | 32x768 | bfloat16 | torch.compile(composite) | 40.64 | 1.58 | 47.9 | n/a |
| rms_norm_fp8_dynamic | 32x1024 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.47 | 2.51 | 40.1 | n/a |
| rms_norm_fp8_dynamic | 32x1024 | bfloat16 | eager composite (norm; amax; quant) | 99.47 | 19.89 | 5.1 | n/a |
| rms_norm_fp8_dynamic | 32x1024 | bfloat16 | torch.compile(composite) | 41.16 | 1.59 | 63.1 | n/a |
| rms_norm_fp8_dynamic | 32x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.46 | 3.11 | 129.1 | n/a |
| rms_norm_fp8_dynamic | 32x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.44 | 28.40 | 14.1 | n/a |
| rms_norm_fp8_dynamic | 32x4096 | bfloat16 | torch.compile(composite) | 40.78 | 2.37 | 169.5 | n/a |
| rms_norm_fp8_dynamic | 17x1023 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.38 | 2.79 | 19.5 | n/a |
| rms_norm_fp8_dynamic | 17x1023 | bfloat16 | eager composite (norm; amax; quant) | 101.31 | 22.50 | 2.4 | n/a |
| rms_norm_fp8_dynamic | 17x1023 | bfloat16 | torch.compile(composite) | 41.66 | 1.66 | 32.8 | n/a |
| rms_norm_fp8_dynamic | 128x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.47 | 3.43 | 461.3 | n/a |
| rms_norm_fp8_dynamic | 128x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.28 | 31.49 | 50.2 | n/a |
| rms_norm_fp8_dynamic | 128x4096 | bfloat16 | torch.compile(composite) | 41.25 | 2.67 | 593.3 | n/a |
| rms_norm_fp8_dynamic | 512x1024 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.42 | 3.42 | 461.5 | n/a |
| rms_norm_fp8_dynamic | 512x1024 | bfloat16 | eager composite (norm; amax; quant) | 98.79 | 24.78 | 63.6 | n/a |
| rms_norm_fp8_dynamic | 512x1024 | bfloat16 | torch.compile(composite) | 41.18 | 2.53 | 623.3 | n/a |
| rms_norm_fp8_dynamic | 1x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.32 | 2.98 | 6.9 | n/a |
| rms_norm_fp8_dynamic | 1x4096 | bfloat16 | eager composite (norm; amax; quant) | 98.62 | 18.60 | 1.1 | n/a |
| rms_norm_fp8_dynamic | 1x4096 | bfloat16 | torch.compile(composite) | 38.68 | 2.21 | 9.3 | n/a |
| rms_norm_fp8_dynamic | 16x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.48 | 3.07 | 66.8 | n/a |
| rms_norm_fp8_dynamic | 16x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.63 | 27.29 | 7.5 | n/a |
| rms_norm_fp8_dynamic | 16x4096 | bfloat16 | torch.compile(composite) | 41.22 | 2.41 | 84.9 | n/a |
| rms_norm_fp8_dynamic | 64x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.47 | 3.20 | 248.6 | n/a |
| rms_norm_fp8_dynamic | 64x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.94 | 29.30 | 27.1 | n/a |
| rms_norm_fp8_dynamic | 64x4096 | bfloat16 | torch.compile(composite) | 41.03 | 2.54 | 312.7 | n/a |
| rms_norm_fp8_dynamic | 192x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.52 | 4.46 | 531.5 | n/a |
| rms_norm_fp8_dynamic | 192x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.64 | 34.37 | 68.9 | n/a |
| rms_norm_fp8_dynamic | 192x4096 | bfloat16 | torch.compile(composite) | 41.48 | 3.42 | 691.8 | n/a |
| rms_norm_fp8_dynamic | 200x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.46 | 4.46 | 553.2 | n/a |
| rms_norm_fp8_dynamic | 200x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.64 | 34.93 | 70.6 | n/a |
| rms_norm_fp8_dynamic | 200x4096 | bfloat16 | torch.compile(composite) | 41.45 | 3.44 | 717.6 | n/a |
| rms_norm_fp8_dynamic | 255x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.47 | 4.58 | 686.6 | n/a |
| rms_norm_fp8_dynamic | 255x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.83 | 36.57 | 85.9 | n/a |
| rms_norm_fp8_dynamic | 255x4096 | bfloat16 | torch.compile(composite) | 41.16 | 3.55 | 886.4 | n/a |
| rms_norm_fp8_dynamic | 256x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.36 | 4.66 | 677.2 | n/a |
| rms_norm_fp8_dynamic | 256x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.74 | 36.70 | 86.0 | n/a |
| rms_norm_fp8_dynamic | 256x4096 | bfloat16 | torch.compile(composite) | 41.05 | 3.66 | 861.5 | n/a |
| rms_norm_fp8_dynamic | 257x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.52 | 4.65 | 681.7 | n/a |
| rms_norm_fp8_dynamic | 257x4096 | bfloat16 | eager composite (norm; amax; quant) | 100.32 | 36.87 | 85.9 | n/a |
| rms_norm_fp8_dynamic | 257x4096 | bfloat16 | torch.compile(composite) | 41.25 | 3.65 | 868.4 | n/a |
| rms_norm_fp8_dynamic | 64x8192 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.43 | 3.91 | 406.7 | n/a |
| rms_norm_fp8_dynamic | 64x8192 | bfloat16 | eager composite (norm; amax; quant) | 99.34 | 28.14 | 56.5 | n/a |
| rms_norm_fp8_dynamic | 64x8192 | bfloat16 | torch.compile(composite) | 40.99 | 3.51 | 452.6 | n/a |
| rms_norm_fp8_dynamic | 32x16384 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.42 | 5.14 | 312.2 | n/a |
| rms_norm_fp8_dynamic | 32x16384 | bfloat16 | eager composite (norm; amax; quant) | 99.15 | 35.10 | 45.8 | n/a |
| rms_norm_fp8_dynamic | 32x16384 | bfloat16 | torch.compile(composite) | 60.98 | 7.30 | 220.0 | n/a |
| rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 33.06 | 31.50 | 1598.9 | n/a |
| rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | eager composite (norm; amax; quant) | 290.30 | 278.25 | 181.0 | n/a |
| rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | torch.compile(composite) | 41.56 | 34.29 | 1468.7 | n/a |
| rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.53 | 11.30 | 2231.0 | n/a |
| rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | eager composite (norm; amax; quant) | 146.06 | 131.86 | 191.1 | n/a |
| rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | torch.compile(composite) | 41.30 | 11.86 | 2124.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x768 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.95 | 2.61 | 66.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x768 | bfloat16 | eager composite | 107.01 | 19.97 | 8.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x768 | bfloat16 | torch.compile(composite) | 43.60 | 1.59 | 109.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x1024 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.13 | 2.62 | 88.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x1024 | bfloat16 | eager composite | 108.05 | 20.91 | 11.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x1024 | bfloat16 | torch.compile(composite) | 43.69 | 1.60 | 144.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.09 | 3.24 | 285.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x4096 | bfloat16 | eager composite | 108.27 | 29.96 | 30.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x4096 | bfloat16 | torch.compile(composite) | 43.71 | 2.59 | 357.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 17x1023 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.93 | 3.00 | 41.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 17x1023 | bfloat16 | eager composite | 111.24 | 24.20 | 5.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 17x1023 | bfloat16 | torch.compile(composite) | 44.06 | 1.77 | 69.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 128x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.10 | 3.59 | 1025.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 128x4096 | bfloat16 | eager composite | 108.94 | 33.68 | 109.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 128x4096 | bfloat16 | torch.compile(composite) | 44.36 | 2.92 | 1259.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 512x1024 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.98 | 4.06 | 905.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 512x1024 | bfloat16 | eager composite | 107.64 | 26.37 | 139.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 512x1024 | bfloat16 | torch.compile(composite) | 44.90 | 2.64 | 1391.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 1x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.95 | 3.03 | 12.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 1x4096 | bfloat16 | eager composite | 108.16 | 19.58 | 1.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 1x4096 | bfloat16 | torch.compile(composite) | 42.82 | 2.39 | 15.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 16x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.05 | 3.16 | 147.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 16x4096 | bfloat16 | eager composite | 108.83 | 28.96 | 16.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 16x4096 | bfloat16 | torch.compile(composite) | 44.43 | 2.67 | 174.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.11 | 3.32 | 555.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x4096 | bfloat16 | eager composite | 108.81 | 31.20 | 59.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x4096 | bfloat16 | torch.compile(composite) | 45.66 | 2.66 | 693.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 192x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.12 | 4.49 | 1227.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 192x4096 | bfloat16 | eager composite | 108.73 | 36.55 | 150.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 192x4096 | bfloat16 | torch.compile(composite) | 45.19 | 3.87 | 1424.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 200x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.11 | 4.54 | 1265.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 200x4096 | bfloat16 | eager composite | 108.74 | 37.00 | 155.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 200x4096 | bfloat16 | torch.compile(composite) | 45.07 | 3.99 | 1439.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 255x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.95 | 4.67 | 1568.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 255x4096 | bfloat16 | eager composite | 108.74 | 39.07 | 187.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 255x4096 | bfloat16 | torch.compile(composite) | 44.45 | 4.06 | 1805.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 256x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.18 | 5.15 | 1426.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 256x4096 | bfloat16 | eager composite | 110.43 | 39.34 | 186.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 256x4096 | bfloat16 | torch.compile(composite) | 46.16 | 4.07 | 1807.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 257x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.17 | 5.10 | 1447.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 257x4096 | bfloat16 | eager composite | 111.63 | 39.31 | 187.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 257x4096 | bfloat16 | torch.compile(composite) | 46.63 | 4.02 | 1836.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x8192 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.12 | 4.45 | 828.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x8192 | bfloat16 | eager composite | 109.91 | 30.70 | 120.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x8192 | bfloat16 | torch.compile(composite) | 47.51 | 4.05 | 909.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x16384 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 18.00 | 6.39 | 579.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x16384 | bfloat16 | eager composite | 108.17 | 38.46 | 96.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x16384 | bfloat16 | torch.compile(composite) | 64.29 | 7.96 | 465.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 45.05 | 43.45 | 2703.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | eager composite | 320.99 | 314.96 | 373.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | torch.compile(composite) | 47.51 | 47.88 | 2453.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 24.30 | 22.42 | 2620.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | eager composite | 159.02 | 148.39 | 396.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | torch.compile(composite) | 44.89 | 18.45 | 3185.2 | n/a |
