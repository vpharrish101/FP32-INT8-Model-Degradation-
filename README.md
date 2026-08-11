# FP32 vs INT8 Evaluation Report

## Experiment Details: -

### 1. Model and Evaluation:

- YOLO11n exported to OpenVINO FP32 and static INT8.
- I evaluated both models on the same 500-image COCO subset.
- Evaluation focused on person, bicycle, car, traffic light, and stop sign (as intended).
- Reported mAP, per-class AP, object-size AP, latency, FPS, and model size.
- Measured single thread latency on CPU (we did this by setting the env var `INFERENCE_NUM_THREADS` to 1).

### 2. Degradation Testing:

- We sampled the same 500 images used for original testing, then degraded them accordingly.

| **Degradation** | **Configuration** |
|---|---|
| Motion blur | 15×15 horizontal averaging kernel |
| Low light | γ = 2.0 |
| JPEG | Quality 30 |
| Downscale | 50% → original resolution |

- The results were recorded accordingly, using both the FP32 and INT8 quantized model.

### 3. Targeted Intervention:

- Motion blur produced the largest accuracy loss, so I selected it for Part 3. I recalibrated INT8 using 1,000 images containing clean and multiple motion-blur strengths: 5×5, 9×9, 15×15, 21×21, and 31×31.

---

## Results: -

### 1. Clean Evaluation

#### Accuracy

| Metric | FP32 | INT8 | Δ |
|---|---:|---:|---:|
| mAP@0.5:0.95 | 0.378 | 0.379 | +0.001 |
| mAP@0.5 | 0.529 | 0.520 | -0.009 |
| AP@0.75 | 0.386 | 0.389 | +0.003 |
| Small | 0.128 | 0.121 | -0.007 |
| Medium | 0.470 | 0.491 | +0.021 |
| Large | 0.699 | 0.702 | +0.003 |

#### Per-Class AP

| Class | FP32 | INT8 | Δ |
|---|---:|---:|---:|
| person | 0.464 | 0.460 | -0.004 |
| bicycle | 0.311 | 0.326 | +0.015 |
| car | 0.326 | 0.323 | -0.003 |
| traffic light | 0.211 | 0.213 | +0.002 |
| stop sign | 0.579 | 0.574 | -0.005 |

#### CPU Latency

**CPU:** Intel Core i7-13650HX  
**Runtime:** OpenVINO CPU, single-threaded

| Metric | FP32 | INT8 |
|---|---:|---:|
| Mean latency | 18.66 ms | 10.99 ms |
| P95 latency | 19.74 ms | 12.18 ms |
| FPS | 53.59 | 90.98 |

INT8 provides approximately **1.70x higher throughput**.

#### Model Size

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

### Accuracy Drop From Each Model's Clean Baseline

| Degradation | FP32 Δ | INT8 Δ |
|---|---:|---:|
| Motion blur | -0.306 | **-0.300** |
| Low light | -0.028 | **-0.022** |
| JPEG Q30 | -0.068 | **-0.066** |
| 50% down → up | -0.058 | **-0.043** |

INT8 does not seem to lose more accuracy under degradation than FP32. The mAP@0.5 drop from the clean baseline was actually smaller for INT8 in all four cases: 0.300 vs 0.306 for motion blur, 0.022 vs 0.028 for low light, 0.066 vs 0.068 for JPEG, and 0.043 vs 0.058 for downscaling. 

This shows that the main metrics loss is due from the degradation itself, which meddles with or removes visual information that the model relies on, rather than from the INT8 quantization. The small differences between FP32 and INT8 are much smaller than the accuracy losses caused by the degradations themselves, and are statistically less significant as they're sub point differences.


---

## 3. Targeted Intervention: Motion Blur Calibration

Motion blur was selected as the worst degradation based on the largest mAP@0.5 loss.

### Motion Blur Results

| Metric | FP32 | Original INT8 | Motion-Calibrated INT8 |
|---|---:|---:|---:|
| mAP@0.5:0.95 | 0.156 | 0.153 | **0.158** |
| mAP@0.5 | 0.223 | 0.220 | **0.225** |
| AP@0.75 | 0.172 | 0.167 | **0.177** |
| Small | 0.003 | 0.003 | 0.003 |
| Medium | 0.176 | 0.166 | **0.181** |
| Large | 0.387 | 0.376 | **0.383** |

### Per-Class AP

| Class | FP32 | Original INT8 | Motion-Calibrated INT8 |
|---|---:|---:|---:|
| person | 0.219 | 0.212 | **0.214** |
| bicycle | 0.022 | 0.024 | 0.022 |
| car | 0.096 | 0.096 | **0.099** |
| traffic light | 0.053 | 0.040 | **0.055** |
| stop sign | 0.389 | 0.392 | **0.399** |

### Intervention Cost

| Metric | Original INT8 | Motion-Calibrated INT8 | Δ |
|---|---:|---:|---:|
| Mean latency | 10.47 ms | 10.34 ms | -0.13 ms |
| P95 latency | 11.38 ms | 11.33 ms | -0.05 ms |
| FPS | 95.48 | 96.75 | +1.27 |
| Model size | 3.3 MB | 3.3 MB | 0 MB |

Motion-blur calibration improved mAP@0.5 from **0.220 to 0.225** and mAP@0.5:0.95 from **0.153 to 0.158**, with no measurable model-size or latency penalty.

The improvement is small, representing partial recovery rather than a complete solution.

---

## Takeaways: -

INT8 did not lose more accuracy than FP32 under any of the four tested degradations. Motion blur was also substantially more damaging than the other degradations for both models. I initially theorized that INT8 would perform slightly worse than FP32, but the evidence proved counterwise. My hypothesis is that, INT8 quantized model has an indirect regularization factor that makes it more or less robust to images, hence the small gap in similarities.

The motion-blur calibration improved the INT8 model, but only by a small amount despite using multiple blur strengths in the calibration set. I hypothezie this as an issue of data processing and not a model centric issue, as: -
  1. INT8's calibration only changed activation values
  2. The blur actively disrupted the information present in the data to be able to make a good prediction.
  3. Hence, this points more towards data/representation bottleneck, not a model issue.

---

## Future Directions: -

- Investigate layer-wise sensitivity to determine whether mixed-precision quantization could recover more of the motion-blur accuracy loss.
- Develop a pre-processing CNN model that is trained on pairs of images and blurred images, to reduce blur in images before letting the models predict. 
- Repeat the evaluation across multiple random 500-image subsets to determine whether the observed improvements are statistically consistent.
- Compare different calibration-set sizes and compositions to determine how much blur data is actually required.
