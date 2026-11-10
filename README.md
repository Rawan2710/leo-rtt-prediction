# RTT-Based Congestion and Handover Prediction for LEO Satellite Networks

This repository contains the source code for the paper:

> R. Khanfar, B. Al Homssi, and M. Saad, "RTT-Based Congestion and Handover Prediction for LEO Satellite Networks," in Proc. BAI, 2027.

## Overview

A two-stage prediction framework that forecasts short-term RTT and classifies upcoming events into Normal, Handover, or Congestion in LEO satellite networks.

- **Stage 1:** PatchTST encoder for RTT forecasting and event probability gating
- **Stage 2:** XGBoost classifier for three-class event classification

## Dataset

This work uses the [LENS OneWeb dataset](https://github.com/clarkzjw/LENS), a LEO satellite network measurement dataset collected from a residential OneWeb terminal deployed in Iowa (March–May 2025). The dataset provides modem-level RTT measurements at 10 ms resolution and Antenna Interface Module (AIM) logs.

## Pipeline

| Script | Description |
|--------|-------------|
| `PatchTST.ipynb` | PatchTST model training |
| `Classifier_Evaluation.py` | XGBoost three-class classifier + end-to-end evaluation |

## Requirements

- Python 3.10+
- PyTorch
- XGBoost
- NumPy, Pandas, Scikit-learn

## Citation

```bibtex
@inproceedings{khanfar2027rtt,
  title={RTT-Based Congestion and Handover Prediction for LEO Satellite Networks},
  author={Khanfar, Rawan and Al Homssi, Bassel and Saad, Mohamed},
  booktitle={Proc. BAI},
  year={2027}
}
```

## License

MIT
