# FP32 vs INT8 — Evaluation Report

## 1. Clean Evaluation

### Accuracy

| Metric | FP32 | INT8 | Δ |
|---|---:|---:|---:|
| mAP@0.5:0.95 | 0.378 | 0.379 | +0.001 |
| mAP@0.5 | 0.529 | 0.520 | -0.009 |
| AP@0.75 | 0.386 | 0.389 | +0.003 |
| Small | 0.128 | 0.121 | -0.007 |
| Medium | 0.470 | 0.491 | +0.021 |
| Large | 0.699 | 0.702 | +0.003 |

### Per-Class AP

| Class | FP32 | INT8 | Δ |
|---|---:|---:|---:|
| person | 0.464 | 0.460 | -0.004 |
| bicycle | 0.311 | 0.326 | +0.015 |
| car | 0.326 | 0.323 | -0.003 |
| traffic light | 0.211 | 0.213 | +0.002 |
| stop sign | 0.579 | 0.574 | -0.005 |

### CPU Latency

**CPU:** Intel Core i7-13650HX  
**Runtime:** OpenVINO CPU, single-threaded

| Metric | FP32 | INT8 |
|---|---:|---:|
| Mean latency | 18.66 ms | 10.99 ms |
| P95 latency | 19.74 ms | 12.18 ms |
| FPS | 53.59 | 90.98 |

INT8 gives approximately **1.70× lower mean latency**.

### Model Size

| Model | Size |
|---|---:|
| FP32 OpenVINO | 10.4 MB |
| INT8 OpenVINO | 3.3 MB |

INT8 is approximately **68% smaller** on disk.

---

## 2. Degradation Evaluation

The same 500 images were used for every degradation.

| Condition | FP32 mAP@0.5 | INT8 mAP@0.5 |
|---|---:|---:|
| Clean | 0.529 | 0.520 |
| Motion blur | 0.223 | 0.220 |
| Low light (γ=2.0) | 0.501 | 0.498 |
| JPEG Q30 | 0.461 | 0.454 |
| 50% downscale → upscale | 0.471 | 0.477 |

### Degradation Parameters

- **Motion blur:** 15×15 horizontal averaging kernel
- **Low light:** gamma correction, γ = 2.0
- **JPEG:** quality = 30
- **Downscale:** 50% of original dimensions using `INTER_AREA`, then restored using `INTER_LINEAR`

### Accuracy Drop From Each Model's Clean Baseline

| Degradation | FP32 Δ | INT8 Δ |
|---|---:|---:|
| Motion blur | -0.306 | **-0.300** |
| Low light | -0.028 | **-0.022** |
| JPEG Q30 | -0.068 | **-0.066** |
| 50% down → up | -0.058 | **-0.043** |

## 3. Conclusion

INT8 did **not** lose more accuracy than FP32 under the tested degradations.

INT8 had a smaller mAP@0.5 drop from its own clean baseline in **all four degradation tests**. The results therefore provide no evidence that INT8 makes the detector more sensitive to these perturbations.

The results are **consistent with a possible regularization-like effect from quantization**, but they do not establish that quantization is the cause. The large common accuracy losses, particularly under motion blur, indicate that the degradation itself is the dominant source of accuracy loss in this experiment.
