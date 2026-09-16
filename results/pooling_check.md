# Pooling check: is the orthogonality null a pooling artefact?

- adapters: `/home/vibe12/projects/persona-curvature/drift/adapters`; 252 modules, 36 layers
- `dW = (lora_alpha/r) * B @ A`; scaling read from adapter_config.json = plain=2, neutral=2, syc_pure=2 (alpha/r, r=16, alpha=32)
- U = dW(plain); V_syc = dW(syc_pure); V_oracle = dW(plain) - dW(neutral)
- random control seed: `20260812`
- module set identical across all adapters: YES

## Sanity check (factored inner product vs dense)

```
[check 1] general_gram vs dense, ranks [4, 6, 3, 5], shape (13,9): max|abs err| = 4.547e-13, rel = 2.421e-16  -> PASS
[check 2] general_gram vs weight_analysis.module_gram (equal rank 5): max|abs err| = 4.547e-13, rel = 1.453e-16  -> PASS
[check 3] cos(U, U-N): dense = 0.964193059412524, factored = 0.964193059412524, |diff| = 1.110e-16  -> PASS
[check 4] global pooled cosine over 3 modules: concat-dense = 0.033872237552441, sum-of-Grams = 0.033872237552441, |diff| = 6.939e-18  -> PASS
[sanity] ALL CHECKS PASSED
```

## A. Global pooled energy fraction (what the original constraint used)

| direction | pooled cos | pooled energy frac | per-module removable frac | x gain (per-mod/pooled) |
|-----------|------------|--------------------|---------------------------|-------------------------|
| V_syc     | 0.2789     | 7.777%             | 8.676%                    | 1.1156                  |
| V_oracle  | 0.7634     | 58.285%            | 58.571%                   | 1.0049                  |
| C_syc     | -0.0000    | 0.000%             | 0.000%                    | 2526.8279               |
| C_oracle  | 0.0000     | 0.000%             | 0.000%                    | 138.1415                |

`per-module removable frac` = sum_m <U_m,V_m>^2/||V_m||^2 / sum_m ||U_m||^2: the energy a PER-MODULE orthogonality constraint would strip, versus the pooled column which is what a single global scalar strips.

## B. Distribution of per-module energy fractions e_m = cos(U_m,V_m)^2

| direction | min     | median  | mean    | p90     | p99     | max     |
|-----------|---------|---------|---------|---------|---------|---------|
| V_syc     | 0.099%  | 10.133% | 10.721% | 17.027% | 29.345% | 40.408% |
| V_oracle  | 27.616% | 55.236% | 54.895% | 62.500% | 69.581% | 70.329% |
| C_syc     | 0.000%  | 0.000%  | 0.000%  | 0.000%  | 0.001%  | 0.003%  |
| C_oracle  | 0.000%  | 0.000%  | 0.000%  | 0.000%  | 0.001%  | 0.001%  |

### top-20 modules by energy fraction, V_syc

| #  | module                     | type      | layer | cos    | energy frac | ||U_m|| | ||V_m|| |
|----|----------------------------|-----------|-------|--------|-------------|---------|---------|
| 1  | layers.4.mlp.up_proj       | up_proj   | 4     | 0.6357 | 40.408%     | 0.5368  | 0.4217  |
| 2  | layers.2.mlp.down_proj     | down_proj | 2     | 0.6125 | 37.514%     | 0.2494  | 0.2093  |
| 3  | layers.3.mlp.down_proj     | down_proj | 3     | 0.5585 | 31.190%     | 0.2341  | 0.2011  |
| 4  | layers.12.self_attn.k_proj | k_proj    | 12    | 0.5251 | 27.573%     | 0.0695  | 0.0620  |
| 5  | layers.2.mlp.up_proj       | up_proj   | 2     | 0.5126 | 26.272%     | 0.3311  | 0.4524  |
| 6  | layers.21.self_attn.v_proj | v_proj    | 21    | 0.4974 | 24.741%     | 0.0901  | 0.0729  |
| 7  | layers.25.self_attn.v_proj | v_proj    | 25    | 0.4967 | 24.674%     | 0.0943  | 0.0755  |
| 8  | layers.20.self_attn.v_proj | v_proj    | 20    | 0.4937 | 24.375%     | 0.0796  | 0.0689  |
| 9  | layers.2.mlp.gate_proj     | gate_proj | 2     | 0.4903 | 24.037%     | 0.3209  | 0.4685  |
| 10 | layers.4.mlp.gate_proj     | gate_proj | 4     | 0.4843 | 23.454%     | 0.5731  | 0.3945  |
| 11 | layers.24.self_attn.v_proj | v_proj    | 24    | 0.4646 | 21.583%     | 0.0913  | 0.0741  |
| 12 | layers.19.self_attn.v_proj | v_proj    | 19    | 0.4605 | 21.202%     | 0.0830  | 0.0646  |
| 13 | layers.5.mlp.gate_proj     | gate_proj | 5     | 0.4592 | 21.090%     | 0.5308  | 0.4114  |
| 14 | layers.17.self_attn.v_proj | v_proj    | 17    | 0.4528 | 20.507%     | 0.0871  | 0.0709  |
| 15 | layers.7.mlp.up_proj       | up_proj   | 7     | 0.4423 | 19.561%     | 0.5045  | 0.3874  |
| 16 | layers.6.mlp.up_proj       | up_proj   | 6     | 0.4351 | 18.928%     | 0.4726  | 0.3781  |
| 17 | layers.12.self_attn.q_proj | q_proj    | 12    | 0.4340 | 18.835%     | 0.2065  | 0.1689  |
| 18 | layers.27.self_attn.k_proj | k_proj    | 27    | 0.4296 | 18.453%     | 0.0882  | 0.0709  |
| 19 | layers.11.self_attn.k_proj | k_proj    | 11    | 0.4284 | 18.357%     | 0.0732  | 0.0658  |
| 20 | layers.27.self_attn.v_proj | v_proj    | 27    | 0.4254 | 18.096%     | 0.0914  | 0.0688  |

### top-20 modules by energy fraction, V_oracle

| #  | module                     | type      | layer | cos    | energy frac | ||U_m|| | ||V_m|| |
|----|----------------------------|-----------|-------|--------|-------------|---------|---------|
| 1  | layers.33.mlp.gate_proj    | gate_proj | 33    | 0.8386 | 70.329%     | 0.7756  | 0.8829  |
| 2  | layers.3.mlp.gate_proj     | gate_proj | 3     | 0.8347 | 69.678%     | 0.5932  | 0.6091  |
| 3  | layers.15.self_attn.k_proj | k_proj    | 15    | 0.8344 | 69.617%     | 0.0841  | 0.1024  |
| 4  | layers.32.mlp.gate_proj    | gate_proj | 32    | 0.8339 | 69.547%     | 0.7735  | 0.8848  |
| 5  | layers.19.self_attn.k_proj | k_proj    | 19    | 0.8269 | 68.369%     | 0.0862  | 0.1032  |
| 6  | layers.35.self_attn.k_proj | k_proj    | 35    | 0.8259 | 68.203%     | 0.1124  | 0.1217  |
| 7  | layers.3.mlp.up_proj       | up_proj   | 3     | 0.8256 | 68.154%     | 0.5904  | 0.6162  |
| 8  | layers.34.mlp.gate_proj    | gate_proj | 34    | 0.8238 | 67.872%     | 0.7821  | 0.8991  |
| 9  | layers.28.mlp.gate_proj    | gate_proj | 28    | 0.8166 | 66.690%     | 0.7650  | 0.8757  |
| 10 | layers.13.self_attn.q_proj | q_proj    | 13    | 0.8164 | 66.659%     | 0.2280  | 0.2464  |
| 11 | layers.4.mlp.gate_proj     | gate_proj | 4     | 0.8096 | 65.538%     | 0.5731  | 0.5514  |
| 12 | layers.35.mlp.gate_proj    | gate_proj | 35    | 0.8089 | 65.430%     | 0.7595  | 0.8760  |
| 13 | layers.29.mlp.gate_proj    | gate_proj | 29    | 0.8088 | 65.408%     | 0.7563  | 0.8607  |
| 14 | layers.27.mlp.gate_proj    | gate_proj | 27    | 0.8081 | 65.296%     | 0.7585  | 0.8684  |
| 15 | layers.30.mlp.gate_proj    | gate_proj | 30    | 0.8047 | 64.757%     | 0.7381  | 0.8516  |
| 16 | layers.26.mlp.gate_proj    | gate_proj | 26    | 0.7999 | 63.978%     | 0.7051  | 0.8123  |
| 17 | layers.35.self_attn.q_proj | q_proj    | 35    | 0.7995 | 63.927%     | 0.3005  | 0.3346  |
| 18 | layers.34.mlp.up_proj      | up_proj   | 34    | 0.7978 | 63.646%     | 0.7607  | 0.8680  |
| 19 | layers.33.mlp.up_proj      | up_proj   | 33    | 0.7963 | 63.403%     | 0.7472  | 0.8661  |
| 20 | layers.31.mlp.gate_proj    | gate_proj | 31    | 0.7946 | 63.136%     | 0.7180  | 0.8198  |

## C. THE DECIDING NUMBER: concentration of trait-aligned energy

| direction               | top5    | top10   | top25   | x uni(5) | x uni(10) | x uni(25) | eff. #modules (PR) |
|-------------------------|---------|---------|---------|----------|-----------|-----------|--------------------|
| V_syc                   | 12.325% | 20.304% | 39.269% | 6.2120   | 5.1165    | 3.9583    | 93.7530            |
| C_syc                   | 18.907% | 28.887% | 50.826% | 9.5289   | 7.2794    | 5.1233    | 58.8732            |
| V_oracle                | 10.385% | 19.757% | 43.019% | 5.2340   | 4.9787    | 4.3363    | 90.2541            |
| C_oracle                | 15.336% | 26.821% | 50.959% | 7.7293   | 6.7589    | 5.1366    | 69.9468            |
| size_baseline_||U_m||^2 | 8.950%  | 17.525% | 39.901% | 4.5109   | 4.4163    | 4.0221    | 98.3858            |

uniform prediction: top5 = 1.984%, top10 = 3.968%, top25 = 9.921%; uniform PR = 252.

| direction | top5 real/ctrl | top10 real/ctrl | top25 real/ctrl | PR real | PR control |
|-----------|----------------|-----------------|-----------------|---------|------------|
| V_syc     | 0.6519         | 0.7029          | 0.7726          | 93.7530 | 58.8732    |
| V_oracle  | 0.6772         | 0.7366          | 0.8442          | 90.2541 | 69.9468    |

## D. Breakdown by module type and depth (mean per-module energy fraction)

| direction | down_proj | gate_proj | k_proj  | o_proj  | q_proj  | up_proj | v_proj  |
|-----------|-----------|-----------|---------|---------|---------|---------|---------|
| V_syc     | 12.121%   | 7.564%    | 10.218% | 11.112% | 7.624%  | 11.524% | 14.885% |
| V_oracle  | 55.464%   | 59.193%   | 54.216% | 52.390% | 55.101% | 55.761% | 52.139% |
| C_syc     | 0.000%    | 0.000%    | 0.000%  | 0.000%  | 0.000%  | 0.000%  | 0.000%  |
| C_oracle  | 0.000%    | 0.000%    | 0.000%  | 0.000%  | 0.000%  | 0.000%  | 0.000%  |

share of total aligned energy by module type:

| direction | down_proj | gate_proj | k_proj  | o_proj  | q_proj  | up_proj | v_proj  |
|-----------|-----------|-----------|---------|---------|---------|---------|---------|
| V_syc     | 10.472%   | 28.962%   | 0.811%  | 8.267%  | 4.632%  | 45.464% | 1.392%  |
| V_oracle  | 7.498%    | 41.999%   | 0.684%  | 5.843%  | 5.498%  | 37.758% | 0.720%  |
| C_syc     | 3.065%    | 15.876%   | 11.513% | 14.452% | 13.224% | 16.916% | 24.954% |
| C_oracle  | 3.768%    | 14.733%   | 13.396% | 13.096% | 14.155% | 20.690% | 20.162% |

| direction | mean early | mean mid | mean late | share early | share mid | share late | corr(e,layer) |
|-----------|------------|----------|-----------|-------------|-----------|------------|---------------|
| V_syc     | 12.392%    | 11.663%  | 8.108%    | 34.885%     | 31.745%   | 33.370%    | -0.3222       |
| V_oracle  | 52.272%    | 55.851%  | 56.562%   | 21.513%     | 27.711%   | 50.776%    | 0.3364        |
| C_syc     | 0.000%     | 0.000%   | 0.000%    | 36.104%     | 25.885%   | 38.011%    | -0.0811       |
| C_oracle  | 0.000%     | 0.000%   | 0.000%    | 20.263%     | 36.883%   | 42.854%    | 0.0256        |

per-layer mean energy fraction:

| dir (mean e_m %) | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11    | 12    | 13    | 14    | 15    | 16    | 17    | 18    | 19    | 20    | 21    | 22    | 23    | 24    | 25    | 26    | 27    | 28    | 29    | 30    | 31    | 32    | 33    | 34    | 35    |
|------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| V_syc            | 9.89  | 10.32 | 19.02 | 9.50  | 18.80 | 12.73 | 12.52 | 11.60 | 8.59  | 11.01 | 11.91 | 12.81 | 14.71 | 10.79 | 10.57 | 10.36 | 10.72 | 12.89 | 11.95 | 11.99 | 14.43 | 13.12 | 7.43  | 10.99 | 11.44 | 11.22 | 7.61  | 10.72 | 7.20  | 6.45  | 5.30  | 7.92  | 6.97  | 7.07  | 7.72  | 7.68  |
| V_oracle         | 47.87 | 46.10 | 41.98 | 56.18 | 56.48 | 50.33 | 52.99 | 55.79 | 53.00 | 55.15 | 54.74 | 56.65 | 56.57 | 58.06 | 54.68 | 57.93 | 57.77 | 55.12 | 54.23 | 58.69 | 53.79 | 54.83 | 55.29 | 53.25 | 52.12 | 51.73 | 57.68 | 55.99 | 58.86 | 57.83 | 58.38 | 56.82 | 55.94 | 57.67 | 56.86 | 58.85 |
| C_oracle         | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  |

## E. Random control

seed `20260812`. per module, A ~ N(0,1) of shape (rank, in), B ~ N(0,1) of shape (out, rank), then the whole delta rescaled so ||C_m||_F == ||V_m||_F for the matched real direction; rank 16 for C_syc, 32 for C_oracle (V_oracle = plain - neutral has rank <= 32)

| control  | min    | median | mean   | p90    | p99    | max    | pooled |
|----------|--------|--------|--------|--------|--------|--------|--------|
| C_syc    | 0.000% | 0.000% | 0.000% | 0.000% | 0.001% | 0.003% | 0.000% |
| C_oracle | 0.000% | 0.000% | 0.000% | 0.000% | 0.001% | 0.001% | 0.000% |

This is what 'no alignment' looks like at rank 16/32 in these shapes. Every real number above must be read against this row, not against zero.

## F. Is V_oracle's alignment with U an arithmetic artefact?

V_oracle = dW(plain) - dW(neutral) contains U = dW(plain) as a term, so cos(U, V_oracle) is an algebraic function of ||U||, ||N||, <U,N> alone. For two ARBITRARY near-orthogonal equal-norm updates it is ~1/sqrt(2), i.e. an energy fraction of ~50%, with no trait structure required.

| quantity                                               | value    |
|--------------------------------------------------------|----------|
| cos(U, N) pooled                                       | 0.1663   |
| ||N|| / ||U|| pooled                                   | 0.7508   |
| cos(U, U-N) PREDICTED from those two alone             | 0.7634   |
| cos(U, U-N) MEASURED                                   | 0.7634   |
| |predicted - measured| (must be ~0: it is an identity) | 0.00e+00 |
| reference: U _|_ N, equal norms -> cos                 | 0.7071   |
| reference energy fraction                              | 50.000%  |
| measured V_oracle pooled energy fraction               | 58.285%  |
| excess over the 50% no-information baseline            | 8.285%   |

| quantity             | min     | median  | mean    | max     |
|----------------------|---------|---------|---------|---------|
| cos(U_m, N_m)        | -0.0244 | 0.1859  | 0.1920  | 0.6241  |
| cos(N_m, V_oracle_m) | -0.6692 | -0.5197 | -0.5148 | -0.1817 |
| cos(V_syc_m, N_m)    | -0.6473 | 0.0625  | 0.0689  | 0.5144  |

If `cos(N_m, V_oracle_m)` is roughly the negative mirror of `cos(U_m, V_oracle_m)`, the oracle direction does not single out `plain` at all -- it is equally (anti-)aligned with `neutral`, which is what a difference vector between two unrelated updates does.
