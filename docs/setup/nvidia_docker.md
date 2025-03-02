# How to Run Docker Containers with CUDA

## Prerequisites

- Ensure your system has an NVIDIA GPU and the appropriate drivers installed.
- Docker must be installed and running on your system.

## Install NVIDIA Container Toolkit

Follow the official NVIDIA guide to install the container toolkit:

- [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Configure Docker Runtime

Run the following command to configure the NVIDIA runtime for Docker:

```sh
sudo nvidia-ctk runtime configure --runtime=docker
```

Example output:
```
INFO[0000] Loading config from /etc/docker/daemon.json
INFO[0000] Wrote updated config to /etc/docker/daemon.json
INFO[0000] It is recommended that docker daemon be restarted.
```

Restart the Docker service to apply the changes:

```sh
sudo systemctl restart docker
```

## Verify Installation

Run a test container to check if CUDA is accessible within Docker:

```sh
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Expected output (example):

```
Unable to find image 'ubuntu:latest' locally
latest: Pulling from library/ubuntu
5a7813e071bf: Pull complete
Digest: sha256:72297848456d5d37d1262630108ab308d3e9ec7ed1c3286a32fe09856619a782
Status: Downloaded newer image for ubuntu:latest
Sun Mar  2 08:18:20 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.16              Driver Version: 570.86.16      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2060 ...    Off |   00000000:01:00.0  On |                  N/A |
|  0%   38C    P8             17W /  175W |     441MiB /   8192MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

If you see GPU details in the output, CUDA is correctly set up in your Docker runtime.

## Conclusion

Your Docker environment is now configured to run containers with CUDA support.
