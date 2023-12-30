# aida-x-trainer
Local GPU-based training utility for the AIDA-X DSP plug-in for MOD devices

## Background
This was created so that I could train AIDA-X models locally, without requiring the use of Google Collab. I am using Windows with WSL2.

## Pre-requisites
First, install WSL2. Then install an Ubuntu release in WSL2 (I am using Ubuntu 22.04.1 LTS).

Second, install Docker Desktop for Windows and ensure you select WSL2 during the installation process.

Third, assumung you have an NVIDIA GPU (otherwise this doesn't make much sense to try to run locally), install the CUDA toolkit by following the steps here:

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

Now, test that you have GPU acceleration by running this benchmark:

`docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark`

You should see something like this:

```
> Windowed mode
> Simulation data stored in video memory
> Single precision floating point simulation
> 1 Devices used for simulation
GPU Device 0: "Ampere" with compute capability 8.6

> Compute 8.6 CUDA device: [NVIDIA GeForce RTX 3070]
47104 bodies, total time for 10 iterations: 37.394 ms
= 593.347 billion interactions per second
= 11866.937 single-precision GFLOP/s at 20 flops per interaction
```

## Running It Locally

First build the Docker from within WSL2 by running:

`docker build -f Dockerfile_pytorch --build-arg host_uid=1000 --build-arg host_gid=1000 . -t pytorch`

Then run the Docker with:

`docker run --gpus all --entrypoint /bin/bash -v $PWD:/workdir:rw -w /workdir -it -p '127.0.0.1:8888:8888' pytorch:latest`

Copy your `input.wav` and `target.wav` files into the `input_files` folder. Optionally open `train_model.py` and update the settings.

Run the script:

`python train_model.py`

After some time (about 8 or 9 minutes on my machine with an RTX 3070 GPU), your model file will be done.
