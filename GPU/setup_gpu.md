# How to install CUDA & cuDNN on Ubuntu 22.04

## Install NVIDIA drivers

### Update & upgrade
```shell
sudo apt update && sudo apt upgrade
```

### Remove previous NVIDIA installation
```shell
sudo apt autoremove nvidia* --purge
```

### Check Ubuntu devices
```shell
ubuntu-drivers devices
```
You will install the NVIDIA driver whose version is tagged with __recommended__


### Install Ubuntu drivers
```shell
sudo ubuntu-drivers autoinstall
```

### Install NVIDIA drivers
My __recommended__ version is 525, adapt to yours

```shell
sudo apt install nvidia-driver-525
```

### Reboot & Check
```shell
reboot
```
after restart verify that the following command works
```shell
nvidia-smi
```

## Install CUDA drivers

### Update & upgrade
```shell
sudo apt update && sudo apt upgrade
```

### Install CUDA toolkit
```shell
sudo apt install nvidia-cuda-toolkit
```

### Check CUDA install
```shell
nvcc --version
```

## Install cuDNN

### Download cuDNN .deb file
You can download cuDNN file [here](https://developer.nvidia.com/rdp/cudnn-download). You will need an Nvidia account.
Select the cuDNN version for the appropriate CUDA version, which is the version that appears when you run:
```shell
nvcc --version
```

### Install cuDNN
```shell
sudo apt install ./<filename.deb>
sudo cp /var/cudnn-<something>.gpg /usr/share/keyrings/
```

My cuDNN version is 8, adapt the following to your version:

```shell
sudo apt update
sudo apt install libcudnn8
sudo apt install libcudnn8-dev
sudo apt install libcudnn8-samples
```

## Test CUDA on Pytorch


### Create a virtualenv and activate it
```shell
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
virtualenv -p py3.10 venv
source venv/bin/activate
```

### Install pytorch
```shell
pip3 install torch torchvision torchaudio
```

### Open Python and execute a test
```python
import torch
print(torch.cuda.is_available()) # should be True

t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
```
