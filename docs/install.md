# Step-by-step installation instructions

Following https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
pip install -v -e .
```

**f. Install Detectron2 and Timm.**
```shell
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**g. Install spconv, torch-scatter, einops**
```shell
pip install spconv-cu111
pip install torch-scatter
pip install einops
```

**h. Clone RCM-Fusion.**
```
git clone https://github.com/mjseong0414/RCM-Fusion.git
```

**i. Prepare pretrained models.**
```
mkdir ckpts

cd ckpts &
```


# Docker instructions

**a. Docker image download and create container.**
```
sudo docker pull tawn0414/rcmfusion:latest

sudo docker run -it -e DISPLAY=unix$DISPLAY --gpus all --ipc=host -v /home:/home -v /mnt:/mnt -v /media:/media -e XAUTHORITY=/tmp/.docker.xauth --name rcmfusion tawn0414/rcmfusion:latest /bin/bash
```

**b. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
pip install -v -e .
```

**c. Clone RCM-Fusion.**
```
git clone https://github.com/mjseong0414/RCM-Fusion.git
```

**d. Prepare pretrained models.**
```
mkdir ckpts

cd ckpts

# rcm-fusion-r50-icra-final checkpoint (google drive link)
[rcm-fusion](https://drive.google.com/file/d/1K985CxSAnjKMkt1vg53MkkSwTTcnRYV-/view?usp=drive_link)

# rcm-fusion-r101-icra-final checkpoint (google drive link)
https://drive.google.com/file/d/1GO5f9DnJHRltFLWEdgv-owjxKvuXIWv4/view?usp=drive_link

```