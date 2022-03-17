#!/bin/bash

# install nano 
sudo apt-get update
sudo apt-get install nano
echo "GNU nano installed!"

# install python3
sudo apt-get update
sudo apt-get install python python3
echo "python3 installed!"

# install pip3
sudo apt-get update
sudo apt-get install python3-pip
echo "pip3 installed!"

# install virtualenv
sudo apt-get update
sudo apt-get install virtualenv
echo "virtualenv installed"

# install jupyter
sudo apt-get update
sudo apt-get install jupyter
echo "jupyter installed"

# install tensorflow
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools==49.6.0
sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
echo "tensorflow installed"

# install pytorch
pip3 install torch torchvision torchaudio
echo "pytorch installed"

# update all
sudo apt-get update
echo "All updated!"

# reboot your system
sudo reboot now
echo "Reboot your system now!"
