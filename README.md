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