FROM gcr.io/kaggle-gpu-images/python@sha256:2e7e3f1b87810248aa8129ce5d7e1d38bd89ae202cc59d29763625abc5bbaba8

# pytorchのバージョンをcuda12.0に対応したものに変える
RUN pip uninstall torch torchvision torchaudio --yes
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

USER root
RUN apt install git --yes

# competition specific