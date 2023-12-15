This is part of my part of 4th Place Solution for the Child Mind Institute - Detect Sleep States (Kaggle competition).
My main part is [here](https://github.com/penguin-prg/child-mind-institute-detect-sleep-states).
detail document: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459597

This repository is based on [this excellent repository](https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states/tree/main) by @tubo213.


# How to Reproduce (for Competition Organizers)
## Hardware
- CPU: Intel Core i9 13900KF (24 cores, 32 threads)
- GPU: NVIDIA GeForce RTX 4090
- RAM: 64GB

## OS/platform
- WSL2 (version 2.0.9.0, Ubuntu 22.04.2 LTS)

## 3rd-party software
Please check the dockerfile in `/kaggle/.devcontainer`

## Training
1. Upload competition dataset in `/kaggle/data`
    - i.e. `/kaggle/data/train_series.parquet`, etc...
2. Run following commands to prepare dataset:
    - `python /kaggle/run/prepare_data.py phase=train`
3. Run follwing commands to train models:
    - `/kaggle/run/train_cv_exp031.sh`
    - `/kaggle/run/train_cv_exp061.sh`

## Supplemental information for hosts
- Dockerfile is used instead of `B4.requirements.txt`.
- `runc/conf` is used instead of `B6. SETTINGS.json`.
- `B7. Serialized copy of the trained model` is [here](https://www.kaggle.com/datasets/ryotayoshinobu/cmi-model).
- `B8. entry_points.md` is not included because all commands are listed above Training section.