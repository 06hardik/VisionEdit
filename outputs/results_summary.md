# VisionEdit -- Architecture Results Summary
*Generated: 2026-09-13 18:05*

## Pipeline Run Summary

| Metric | Value |
|--------|-------|
| Total clips scored | 8 |
| Passed quality gate | 7 / 8 |
| Average S\_i | 0.4674 |
| Top clip (scene 007) | S\_i = 0.7935 (surprise / person) |
| Expert corrections fired | 1 / 8 clips |
| Stage-2 OD triggered | 5 / 8 clips |
| Avg cascade FLOP savings | 43.4% |

## Per-Clip Results Table

| Sc | Start | End | E\_i | Emotion | Intensity | EX | O\_i | TopClass | S2 | Sav% | M\_i | Gate | **S\_i** |
|----|----|-----|------|---------|-----------|----|----|------|----|----|----|------|---------|
| 00 | 0.00 | 3.20 | +0.720 | happy | STRONG | N | 0.840 | person | Y | 69.5 | 0.620 | PASS | **0.7370** |
| 01 | 3.20 | 6.40 | +0.150 | neutral | MINIMAL | N | 0.530 | car | Y | 69.5 | 0.410 | PASS | **0.3480** |
| 02 | 6.40 | 9.10 | +0.310 | disgust | AVERAGE | Y | 0.470 | person | N | 0.0 | 0.280 | PASS | **0.3585** |
| 03 | 9.10 | 11.80 | +0.550 | happy | AVERAGE | N | 0.610 | dog | N | 0.0 | 0.710 | FAIL | **0.0000** |
| 04 | 11.80 | 14.30 | +0.000 | neutral | MINIMAL | N | 0.780 | bicycle | Y | 69.5 | 0.880 | PASS | **0.4930** |
| 05 | 14.30 | 17.50 | +0.480 | surprise | AVERAGE | N | 0.910 | person | Y | 69.5 | 0.550 | PASS | **0.6480** |
| 06 | 17.50 | 19.90 | +0.120 | neutral | MINIMAL | N | 0.660 | car | N | 0.0 | 0.330 | PASS | **0.3615** |
| 07 | 19.90 | 22.60 | +0.830 | surprise | STRONG | N | 0.790 | person | Y | 69.5 | 0.740 | PASS | **0.7935** |

> **Note**: `EX` = binary expert corrected this clip.
> `S2` = YOLOv9-E Stage 2 triggered.
> `Sav%` = estimated cascade FLOP reduction.

## Key Architecture Improvements vs Literature

| Paper | Limitation | Our Improvement |
|-------|-----------|-----------------|
| Punuri 2024 [A3] | LRP fails on misclassified frames | Grad-CAM (all frames) |
| Gursesli 2024 [A1] | disgust/fear F1 ~40% | Binary expert CNNs |
| Abbas 2025 [A2] | Tested on lab-controlled CK+ only | WeightedFERLoss |
| Salman 2025 [A4] | 8 full backbones/frame | 2 lightweight experts <60% conf |
| Kosta 2023 [A5] | No temporal emotion modeling | LSTM rolling-window |
| Shah 2026 [B3] | Fixed tau=75 @ 15fps | Adaptive tau = tau_sec * fps |
| Yang 2024 [B4] | IoU-only label assignment | SaIS = IoU + shape_score |
| Hua 2025 [B1] | Single model every frame | Two-stage cascade ~69.5% savings |

## Grad-CAM Intensity Distribution

| Intensity | Count | % |
|-----------|-------|---|
| MINIMAL | 3 | 37% |
| AVERAGE | 3 | 37% |
| STRONG | 2 | 25% |

> Intensity ranking adapted from Punuri et al. (2024) [A3].
> Driven by Grad-CAM mean activation (our improvement over LRP).
