# bench_norms (2026-10-29T22:33:27+00:00)

NVIDIA H100 NVL, torch 2.13.0+cu130, CUDA 13.0, commit 93d588289b9d034e146a62576308d9e110e80d49 (dirty=True), driver 580.173.02.

| op | MxN | dtype | candidate | eager us | kernel us | GB/s | % peak |
|---|---|---|---|---:|---:|---:|---:|
| rms_norm | 32x768 | bfloat16 | fused_layernorm.rms_norm | 8.96 | 1.84 | 54.2 | n/a |
| rms_norm | 32x768 | bfloat16 | F.rms_norm (aten fused kernel) | 7.33 | 2.18 | 45.8 | n/a |
| rms_norm | 32x768 | bfloat16 | torch.compile(F.rms_norm) | 31.73 | 1.19 | 84.1 | n/a |
| rms_norm | 32x1024 | bfloat16 | fused_layernorm.rms_norm | 11.47 | 1.90 | 69.9 | n/a |
| rms_norm | 32x1024 | bfloat16 | F.rms_norm (aten fused kernel) | 10.83 | 2.19 | 60.7 | n/a |
| rms_norm | 32x1024 | bfloat16 | torch.compile(F.rms_norm) | 35.46 | 1.19 | 112.0 | n/a |
| rms_norm | 32x4096 | bfloat16 | fused_layernorm.rms_norm | 11.60 | 3.28 | 162.3 | n/a |
| rms_norm | 32x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.92 | 4.59 | 116.0 | n/a |
| rms_norm | 32x4096 | bfloat16 | torch.compile(F.rms_norm) | 36.05 | 1.61 | 330.8 | n/a |
| rms_norm | 17x1023 | bfloat16 | fused_layernorm.rms_norm | 11.63 | 1.87 | 38.2 | n/a |
| rms_norm | 17x1023 | bfloat16 | F.rms_norm (aten fused kernel) | 13.34 | 4.76 | 15.0 | n/a |
| rms_norm | 17x1023 | bfloat16 | torch.compile(F.rms_norm) | 35.06 | 1.29 | 55.5 | n/a |
| rms_norm | 128x4096 | bfloat16 | fused_layernorm.rms_norm | 11.31 | 3.83 | 550.2 | n/a |
| rms_norm | 128x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.71 | 4.94 | 426.0 | n/a |
| rms_norm | 128x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.43 | 2.14 | 983.5 | n/a |
| rms_norm | 512x1024 | bfloat16 | fused_layernorm.rms_norm | 11.31 | 2.74 | 767.3 | n/a |
| rms_norm | 512x1024 | bfloat16 | F.rms_norm (aten fused kernel) | 10.71 | 2.82 | 745.1 | n/a |
| rms_norm | 512x1024 | bfloat16 | torch.compile(F.rms_norm) | 36.77 | 1.74 | 1203.6 | n/a |
| rms_norm | 1x4096 | bfloat16 | fused_layernorm.rms_norm | 11.34 | 3.07 | 8.0 | n/a |
| rms_norm | 1x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.65 | 4.42 | 5.6 | n/a |
| rms_norm | 1x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.80 | 1.46 | 16.9 | n/a |
| rms_norm | 16x4096 | bfloat16 | fused_layernorm.rms_norm | 11.25 | 3.22 | 84.1 | n/a |
| rms_norm | 16x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.65 | 4.52 | 59.8 | n/a |
| rms_norm | 16x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.60 | 1.77 | 152.3 | n/a |
| rms_norm | 64x4096 | bfloat16 | fused_layernorm.rms_norm | 11.29 | 3.41 | 309.7 | n/a |
| rms_norm | 64x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.62 | 4.67 | 226.1 | n/a |
| rms_norm | 64x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.72 | 1.69 | 623.5 | n/a |
| rms_norm | 192x4096 | bfloat16 | fused_layernorm.rms_norm | 11.23 | 4.02 | 785.4 | n/a |
| rms_norm | 192x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.62 | 5.06 | 623.6 | n/a |
| rms_norm | 192x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.09 | 2.40 | 1311.8 | n/a |
| rms_norm | 200x4096 | bfloat16 | fused_layernorm.rms_norm | 11.23 | 4.05 | 811.7 | n/a |
| rms_norm | 200x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.74 | 5.07 | 648.3 | n/a |
| rms_norm | 200x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.31 | 2.17 | 1512.1 | n/a |
| rms_norm | 255x4096 | bfloat16 | fused_layernorm.rms_norm | 11.25 | 4.16 | 1006.0 | n/a |
| rms_norm | 255x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.69 | 5.18 | 808.5 | n/a |
| rms_norm | 255x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.40 | 2.27 | 1846.9 | n/a |
| rms_norm | 256x4096 | bfloat16 | fused_layernorm.rms_norm | 11.20 | 3.05 | 1378.9 | n/a |
| rms_norm | 256x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.66 | 5.17 | 812.9 | n/a |
| rms_norm | 256x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.35 | 2.26 | 1857.7 | n/a |
| rms_norm | 257x4096 | bfloat16 | fused_layernorm.rms_norm | 11.30 | 2.98 | 1417.4 | n/a |
| rms_norm | 257x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 10.68 | 5.16 | 817.4 | n/a |
| rms_norm | 257x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.15 | 2.59 | 1630.2 | n/a |
| rms_norm | 64x8192 | bfloat16 | fused_layernorm.rms_norm | 11.10 | 5.18 | 408.3 | n/a |
| rms_norm | 64x8192 | bfloat16 | F.rms_norm (aten fused kernel) | 10.55 | 7.96 | 265.4 | n/a |
| rms_norm | 64x8192 | bfloat16 | torch.compile(F.rms_norm) | 34.35 | 2.12 | 996.9 | n/a |
| rms_norm | 32x16384 | bfloat16 | fused_layernorm.rms_norm | 11.22 | 8.59 | 248.1 | n/a |
| rms_norm | 32x16384 | bfloat16 | F.rms_norm (aten fused kernel) | 15.80 | 14.46 | 147.3 | n/a |
| rms_norm | 32x16384 | bfloat16 | torch.compile(F.rms_norm) | 47.30 | 4.34 | 490.2 | n/a |
| rms_norm | 4096x4096 | bfloat16 | fused_layernorm.rms_norm | 23.34 | 21.55 | 3114.4 | n/a |
| rms_norm | 4096x4096 | bfloat16 | F.rms_norm (aten fused kernel) | 25.99 | 24.74 | 2713.4 | n/a |
| rms_norm | 4096x4096 | bfloat16 | torch.compile(F.rms_norm) | 34.71 | 23.99 | 2797.4 | n/a |
| rms_norm | 8192x1024 | bfloat16 | fused_layernorm.rms_norm | 11.35 | 8.51 | 3943.1 | n/a |
| rms_norm | 8192x1024 | bfloat16 | F.rms_norm (aten fused kernel) | 12.77 | 11.64 | 2882.5 | n/a |
| rms_norm | 8192x1024 | bfloat16 | torch.compile(F.rms_norm) | 34.32 | 8.73 | 3844.9 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | fused op | 13.68 | 1.93 | 102.8 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | fused op (inplace) | 11.95 | 1.92 | 103.1 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | eager composite (add; norm) | 17.38 | 3.34 | 59.3 | n/a |
| fused_add_rms_norm | 32x768 | bfloat16 | torch.compile(composite) | 37.81 | 1.23 | 161.7 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | fused op | 13.72 | 2.02 | 130.9 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | fused op (inplace) | 11.90 | 2.01 | 131.3 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | eager composite (add; norm) | 17.38 | 3.23 | 81.9 | n/a |
| fused_add_rms_norm | 32x1024 | bfloat16 | torch.compile(composite) | 37.56 | 1.24 | 213.8 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | fused op | 13.70 | 3.62 | 291.7 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | fused op (inplace) | 11.87 | 3.58 | 294.8 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | eager composite (add; norm) | 17.40 | 6.12 | 172.6 | n/a |
| fused_add_rms_norm | 32x4096 | bfloat16 | torch.compile(composite) | 37.88 | 1.80 | 588.2 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | fused op | 13.73 | 2.03 | 69.5 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | fused op (inplace) | 11.88 | 1.99 | 70.9 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | eager composite (add; norm) | 19.87 | 6.31 | 22.4 | n/a |
| fused_add_rms_norm | 17x1023 | bfloat16 | torch.compile(composite) | 38.19 | 1.44 | 98.3 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | fused op | 13.71 | 4.14 | 1016.1 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | fused op (inplace) | 12.13 | 4.13 | 1017.9 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | eager composite (add; norm) | 17.35 | 6.93 | 606.3 | n/a |
| fused_add_rms_norm | 128x4096 | bfloat16 | torch.compile(composite) | 37.80 | 2.48 | 1697.7 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | fused op | 13.54 | 3.01 | 1396.3 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | fused op (inplace) | 11.95 | 3.01 | 1394.6 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | eager composite (add; norm) | 17.24 | 4.39 | 955.3 | n/a |
| fused_add_rms_norm | 512x1024 | bfloat16 | torch.compile(composite) | 37.62 | 1.93 | 2173.7 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | fused op | 13.95 | 3.43 | 11.9 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | fused op (inplace) | 12.11 | 3.43 | 11.9 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | eager composite (add; norm) | 17.54 | 5.37 | 7.6 | n/a |
| fused_add_rms_norm | 1x4096 | bfloat16 | torch.compile(composite) | 38.18 | 1.63 | 25.1 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | fused op | 13.86 | 3.52 | 151.3 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | fused op (inplace) | 12.09 | 3.50 | 151.9 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | eager composite (add; norm) | 17.66 | 5.98 | 89.1 | n/a |
| fused_add_rms_norm | 16x4096 | bfloat16 | torch.compile(composite) | 38.12 | 1.75 | 304.9 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | fused op | 13.78 | 3.73 | 563.9 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | fused op (inplace) | 11.95 | 3.74 | 562.9 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | eager composite (add; norm) | 17.50 | 6.38 | 330.2 | n/a |
| fused_add_rms_norm | 64x4096 | bfloat16 | torch.compile(composite) | 38.04 | 2.20 | 958.1 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | fused op | 13.63 | 4.65 | 1354.7 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | fused op (inplace) | 11.93 | 4.65 | 1355.0 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | eager composite (add; norm) | 17.38 | 7.32 | 860.1 | n/a |
| fused_add_rms_norm | 192x4096 | bfloat16 | torch.compile(composite) | 37.97 | 2.82 | 2233.4 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | fused op | 13.73 | 4.64 | 1413.4 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | fused op (inplace) | 11.91 | 4.64 | 1413.5 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | eager composite (add; norm) | 17.46 | 7.36 | 892.0 | n/a |
| fused_add_rms_norm | 200x4096 | bfloat16 | torch.compile(composite) | 38.00 | 2.56 | 2563.8 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | fused op | 13.66 | 4.70 | 1778.5 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | fused op (inplace) | 11.89 | 4.69 | 1782.8 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | eager composite (add; norm) | 17.42 | 7.66 | 1091.8 | n/a |
| fused_add_rms_norm | 255x4096 | bfloat16 | torch.compile(composite) | 37.91 | 2.65 | 3154.8 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | fused op | 13.83 | 3.46 | 2429.9 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | fused op (inplace) | 12.10 | 3.46 | 2430.0 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | eager composite (add; norm) | 17.46 | 7.71 | 1088.4 | n/a |
| fused_add_rms_norm | 256x4096 | bfloat16 | torch.compile(composite) | 37.75 | 2.64 | 3177.4 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | fused op | 13.79 | 3.49 | 2418.4 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | fused op (inplace) | 12.06 | 3.49 | 2412.2 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | eager composite (add; norm) | 17.46 | 7.61 | 1107.7 | n/a |
| fused_add_rms_norm | 257x4096 | bfloat16 | torch.compile(composite) | 37.61 | 2.64 | 3196.3 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | fused op | 13.71 | 5.88 | 715.8 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | fused op (inplace) | 12.14 | 5.86 | 718.6 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | eager composite (add; norm) | 17.23 | 10.35 | 406.7 | n/a |
| fused_add_rms_norm | 64x8192 | bfloat16 | torch.compile(composite) | 37.62 | 2.43 | 1734.7 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | fused op | 13.59 | 9.80 | 431.5 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | fused op (inplace) | 11.88 | 9.80 | 431.5 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | eager composite (add; norm) | 20.06 | 17.55 | 240.9 | n/a |
| fused_add_rms_norm | 32x16384 | bfloat16 | torch.compile(composite) | 50.18 | 4.92 | 859.1 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | fused op | 43.13 | 41.76 | 3214.0 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | fused op (inplace) | 42.80 | 44.94 | 2987.0 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | eager composite (add; norm) | 58.98 | 62.34 | 2153.2 | n/a |
| fused_add_rms_norm | 4096x4096 | bfloat16 | torch.compile(composite) | 45.07 | 41.91 | 3202.7 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | fused op | 22.86 | 20.87 | 3216.1 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | fused op (inplace) | 20.19 | 18.88 | 3553.9 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | eager composite (add; norm) | 30.73 | 28.36 | 2366.3 | n/a |
| fused_add_rms_norm | 8192x1024 | bfloat16 | torch.compile(composite) | 37.92 | 19.08 | 3518.2 | n/a |
| rms_norm_fp8_dynamic | 32x768 | bfloat16 | fused rms_norm_fp8 (dynamic) | 13.88 | 2.72 | 27.8 | n/a |
| rms_norm_fp8_dynamic | 32x768 | bfloat16 | eager composite (norm; amax; quant) | 96.69 | 18.43 | 4.1 | n/a |
| rms_norm_fp8_dynamic | 32x768 | bfloat16 | torch.compile(composite) | 40.24 | 1.58 | 47.6 | n/a |
| rms_norm_fp8_dynamic | 32x1024 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.06 | 2.86 | 35.1 | n/a |
| rms_norm_fp8_dynamic | 32x1024 | bfloat16 | eager composite (norm; amax; quant) | 97.97 | 19.59 | 5.1 | n/a |
| rms_norm_fp8_dynamic | 32x1024 | bfloat16 | torch.compile(composite) | 40.36 | 1.60 | 62.8 | n/a |
| rms_norm_fp8_dynamic | 32x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.10 | 4.66 | 86.2 | n/a |
| rms_norm_fp8_dynamic | 32x4096 | bfloat16 | eager composite (norm; amax; quant) | 97.84 | 28.18 | 14.2 | n/a |
| rms_norm_fp8_dynamic | 32x4096 | bfloat16 | torch.compile(composite) | 40.54 | 2.37 | 169.6 | n/a |
| rms_norm_fp8_dynamic | 17x1023 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.21 | 2.78 | 19.5 | n/a |
| rms_norm_fp8_dynamic | 17x1023 | bfloat16 | eager composite (norm; amax; quant) | 100.97 | 22.53 | 2.4 | n/a |
| rms_norm_fp8_dynamic | 17x1023 | bfloat16 | torch.compile(composite) | 41.27 | 1.66 | 32.7 | n/a |
| rms_norm_fp8_dynamic | 128x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.01 | 5.13 | 308.5 | n/a |
| rms_norm_fp8_dynamic | 128x4096 | bfloat16 | eager composite (norm; amax; quant) | 97.49 | 31.66 | 50.0 | n/a |
| rms_norm_fp8_dynamic | 128x4096 | bfloat16 | torch.compile(composite) | 40.51 | 2.66 | 594.4 | n/a |
| rms_norm_fp8_dynamic | 512x1024 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.04 | 3.82 | 413.2 | n/a |
| rms_norm_fp8_dynamic | 512x1024 | bfloat16 | eager composite (norm; amax; quant) | 97.09 | 24.68 | 63.9 | n/a |
| rms_norm_fp8_dynamic | 512x1024 | bfloat16 | torch.compile(composite) | 40.42 | 2.53 | 624.5 | n/a |
| rms_norm_fp8_dynamic | 1x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.03 | 4.47 | 4.6 | n/a |
| rms_norm_fp8_dynamic | 1x4096 | bfloat16 | eager composite (norm; amax; quant) | 97.02 | 18.48 | 1.1 | n/a |
| rms_norm_fp8_dynamic | 1x4096 | bfloat16 | torch.compile(composite) | 38.30 | 2.21 | 9.3 | n/a |
| rms_norm_fp8_dynamic | 16x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.09 | 4.56 | 44.9 | n/a |
| rms_norm_fp8_dynamic | 16x4096 | bfloat16 | eager composite (norm; amax; quant) | 98.41 | 27.08 | 7.6 | n/a |
| rms_norm_fp8_dynamic | 16x4096 | bfloat16 | torch.compile(composite) | 40.83 | 2.42 | 84.8 | n/a |
| rms_norm_fp8_dynamic | 64x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.16 | 4.76 | 167.0 | n/a |
| rms_norm_fp8_dynamic | 64x4096 | bfloat16 | eager composite (norm; amax; quant) | 98.98 | 29.24 | 27.2 | n/a |
| rms_norm_fp8_dynamic | 64x4096 | bfloat16 | torch.compile(composite) | 41.25 | 2.54 | 313.1 | n/a |
| rms_norm_fp8_dynamic | 192x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.24 | 6.19 | 382.9 | n/a |
| rms_norm_fp8_dynamic | 192x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.40 | 34.29 | 69.1 | n/a |
| rms_norm_fp8_dynamic | 192x4096 | bfloat16 | torch.compile(composite) | 40.93 | 3.42 | 692.7 | n/a |
| rms_norm_fp8_dynamic | 200x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.11 | 6.20 | 397.7 | n/a |
| rms_norm_fp8_dynamic | 200x4096 | bfloat16 | eager composite (norm; amax; quant) | 98.89 | 34.86 | 70.8 | n/a |
| rms_norm_fp8_dynamic | 200x4096 | bfloat16 | torch.compile(composite) | 41.04 | 3.43 | 719.1 | n/a |
| rms_norm_fp8_dynamic | 255x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.06 | 6.29 | 499.4 | n/a |
| rms_norm_fp8_dynamic | 255x4096 | bfloat16 | eager composite (norm; amax; quant) | 98.55 | 36.39 | 86.4 | n/a |
| rms_norm_fp8_dynamic | 255x4096 | bfloat16 | torch.compile(composite) | 40.96 | 3.51 | 894.1 | n/a |
| rms_norm_fp8_dynamic | 256x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.09 | 4.65 | 678.3 | n/a |
| rms_norm_fp8_dynamic | 256x4096 | bfloat16 | eager composite (norm; amax; quant) | 99.27 | 36.51 | 86.4 | n/a |
| rms_norm_fp8_dynamic | 256x4096 | bfloat16 | torch.compile(composite) | 40.95 | 3.65 | 863.4 | n/a |
| rms_norm_fp8_dynamic | 257x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.03 | 4.65 | 680.7 | n/a |
| rms_norm_fp8_dynamic | 257x4096 | bfloat16 | eager composite (norm; amax; quant) | 98.53 | 36.66 | 86.4 | n/a |
| rms_norm_fp8_dynamic | 257x4096 | bfloat16 | torch.compile(composite) | 40.74 | 3.65 | 867.8 | n/a |
| rms_norm_fp8_dynamic | 64x8192 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.17 | 7.05 | 225.4 | n/a |
| rms_norm_fp8_dynamic | 64x8192 | bfloat16 | eager composite (norm; amax; quant) | 97.97 | 28.33 | 56.1 | n/a |
| rms_norm_fp8_dynamic | 64x8192 | bfloat16 | torch.compile(composite) | 41.20 | 3.50 | 454.2 | n/a |
| rms_norm_fp8_dynamic | 32x16384 | bfloat16 | fused rms_norm_fp8 (dynamic) | 14.11 | 11.51 | 139.5 | n/a |
| rms_norm_fp8_dynamic | 32x16384 | bfloat16 | eager composite (norm; amax; quant) | 98.14 | 35.11 | 45.7 | n/a |
| rms_norm_fp8_dynamic | 32x16384 | bfloat16 | torch.compile(composite) | 60.61 | 7.34 | 218.8 | n/a |
| rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | fused rms_norm_fp8 (dynamic) | 33.13 | 31.47 | 1600.1 | n/a |
| rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | eager composite (norm; amax; quant) | 290.71 | 277.86 | 181.2 | n/a |
| rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | torch.compile(composite) | 40.78 | 34.02 | 1480.4 | n/a |
| rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | fused rms_norm_fp8 (dynamic) | 18.21 | 16.81 | 1499.1 | n/a |
| rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | eager composite (norm; amax; quant) | 146.64 | 131.57 | 191.5 | n/a |
| rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | torch.compile(composite) | 40.49 | 11.85 | 2126.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x768 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.46 | 2.79 | 62.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x768 | bfloat16 | eager composite | 104.88 | 19.59 | 8.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x768 | bfloat16 | torch.compile(composite) | 43.20 | 1.59 | 109.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x1024 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.62 | 2.95 | 78.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x1024 | bfloat16 | eager composite | 106.42 | 20.88 | 11.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x1024 | bfloat16 | torch.compile(composite) | 43.51 | 1.61 | 143.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.52 | 4.97 | 186.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x4096 | bfloat16 | eager composite | 105.79 | 29.84 | 31.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x4096 | bfloat16 | torch.compile(composite) | 42.87 | 2.60 | 356.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 17x1023 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.51 | 2.98 | 41.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 17x1023 | bfloat16 | eager composite | 109.68 | 23.82 | 5.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 17x1023 | bfloat16 | torch.compile(composite) | 43.04 | 1.78 | 69.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 128x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.81 | 5.42 | 679.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 128x4096 | bfloat16 | eager composite | 108.42 | 33.43 | 110.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 128x4096 | bfloat16 | torch.compile(composite) | 44.54 | 2.93 | 1257.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 512x1024 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.49 | 4.08 | 900.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 512x1024 | bfloat16 | eager composite | 105.98 | 26.36 | 139.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 512x1024 | bfloat16 | torch.compile(composite) | 43.89 | 2.64 | 1392.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 1x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.57 | 4.77 | 7.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 1x4096 | bfloat16 | eager composite | 106.55 | 19.77 | 1.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 1x4096 | bfloat16 | torch.compile(composite) | 41.91 | 2.38 | 15.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 16x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.72 | 4.89 | 95.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 16x4096 | bfloat16 | eager composite | 107.69 | 28.83 | 16.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 16x4096 | bfloat16 | torch.compile(composite) | 43.45 | 2.67 | 175.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.67 | 5.06 | 364.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x4096 | bfloat16 | eager composite | 107.07 | 30.86 | 59.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x4096 | bfloat16 | torch.compile(composite) | 43.61 | 2.66 | 693.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 192x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.61 | 6.79 | 812.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 192x4096 | bfloat16 | eager composite | 106.28 | 36.56 | 150.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 192x4096 | bfloat16 | torch.compile(composite) | 43.25 | 3.91 | 1408.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 200x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.73 | 6.77 | 847.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 200x4096 | bfloat16 | eager composite | 106.87 | 37.12 | 154.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 200x4096 | bfloat16 | torch.compile(composite) | 43.98 | 3.94 | 1458.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 255x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.75 | 6.87 | 1065.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 255x4096 | bfloat16 | eager composite | 108.14 | 39.07 | 187.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 255x4096 | bfloat16 | torch.compile(composite) | 43.67 | 4.08 | 1794.0 | n/a |
| fused_add_rms_norm_fp8_dynamic | 256x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.70 | 5.16 | 1424.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 256x4096 | bfloat16 | eager composite | 107.96 | 39.07 | 188.1 | n/a |
| fused_add_rms_norm_fp8_dynamic | 256x4096 | bfloat16 | torch.compile(composite) | 44.14 | 4.09 | 1797.8 | n/a |
| fused_add_rms_norm_fp8_dynamic | 257x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.51 | 5.10 | 1445.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 257x4096 | bfloat16 | eager composite | 107.53 | 39.27 | 187.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 257x4096 | bfloat16 | torch.compile(composite) | 43.92 | 4.04 | 1827.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x8192 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.68 | 7.67 | 480.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x8192 | bfloat16 | eager composite | 106.65 | 30.87 | 119.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 64x8192 | bfloat16 | torch.compile(composite) | 43.36 | 4.06 | 908.4 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x16384 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 17.56 | 12.70 | 291.7 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x16384 | bfloat16 | eager composite | 105.43 | 38.45 | 96.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 32x16384 | bfloat16 | torch.compile(composite) | 63.10 | 7.99 | 463.3 | n/a |
| fused_add_rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 45.09 | 43.40 | 2706.6 | n/a |
| fused_add_rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | eager composite | 321.58 | 314.75 | 373.2 | n/a |
| fused_add_rms_norm_fp8_dynamic | 4096x4096 | bfloat16 | torch.compile(composite) | 47.86 | 47.67 | 2463.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | fused_add_rms_norm_fp8 (dynamic) | 26.24 | 24.66 | 2382.9 | n/a |
| fused_add_rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | eager composite | 160.42 | 147.80 | 397.5 | n/a |
| fused_add_rms_norm_fp8_dynamic | 8192x1024 | bfloat16 | torch.compile(composite) | 43.94 | 18.51 | 3173.6 | n/a |
