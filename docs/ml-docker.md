# install pip3 (on Ubuntu)

        pip3 install python3-pip

# install docker

- Check your suitable docker version

        $ apt cache madison docker-ce

- Remove any Docker files that are running in the system, using the following command

        $ sudo apt-get remove docker docker-engine docker.io

- Check if the system is up-to-date using the following command

        $ sudo apt-get update

- Install Docker using the following command

        $ sudo apt install docker.io

- Install all the dependency packages using the following command (You’ll then get a prompt asking you to choose between y/n - choose y)

        $ sudo snap install docker

- Before testing Docker, check the version installed using the following command

        $ docker version
        $ docker --version

- Pull an image from the Docker hub using the following command

        $ sudo docker images

# working with docker

- load image tar archive file

        $ docker load -i <path/to/docker/file.tar>

- build container from image

        $ docker run -t --name <docker container name> -v$PWD/your/folder:/<container-folder> <your image>:<tag>

- start docker image

        $ docker start <docker container name>

- jump in docker container

        $ docker exec -it <your container name> bash

- run the docker container with specific command outside the container

        $ docker exec -it <your container name> <command>

- check all docker image and docker container

        $ docker system df

- delete all docker images

        $ sudo docker system prune -a

- check all docker container

        $ docker container ps -a

- stop docker container

        $ docker stop <your container name>|<your container id>

- remove docker container

        $ docker rm <your container name>|<your container id>

- delete docker image

        $ docker rmi <your image name>:<tag>

# build and run GPU accelerated Docker containers

[user-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)

[Enabling GPU access with Compose](https://docs.docker.com/compose/gpu-support/)

[github](https://github.com/NVIDIA/nvidia-docker)

[NVIDIA Driver Installation](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)

[docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

[cuda version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

## install docker

## install nvidia docker

[Setting up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

Setup the package repository and the GPG key:

    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    $ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    $ curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

Install the nvidia-docker2 package (and dependencies) after updating the package listing:

    $ sudo apt-get update
    $ sudo apt-get install -y nvidia-docker2

Restart the dokcer deamon to complete the installation after setting default runtime

    $ sudo systemctl restart docker

At this point, a working setup can be tested by running a base CUDA container

    $ sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

## using docker hub to pulling tensorflow image

[tensorflow/tensorflow image docer hub](https://hub.docker.com/r/tensorflow/tensorflow)

    $ docker pull tensorflow/tensorflow                     # latest stable release
    $ docker pull tensorflow/tensorflow:devel-gpu           # nightly dev release w/ GPU support
    $ docker pull tensorflow/tensorflow:latest-gpu-jupyter  # latest release w/ GPU support and Jupyter

## running containers

run docker container, jump in and remove the container after done

    $ docker run -it --rm gpus all -v $(pwd):/workspace/ tensorflow/tensorflow:nightly-gpu bash

run and keep the container

    $ docker run -it gpus all -v $(pwd):/workspace/ tensorflow/tensorflow:nightly-gpu bash

## deleting containers
