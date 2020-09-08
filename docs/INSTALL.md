## Install

![Python](https://img.shields.io/badge/Python->=3.7-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.3.0-Orange?logo=pytorch) ![mmcv](https://img.shields.io/badge/mmcv-%3E%3D0.5.0-green)


1. Install requirements

    ```pip install -r requirements.txt```

2. Install pytorch 

    ```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```

    Make sure the cudatoolkit version is the same as your system's CUDA. Here we use CUDA 10.2. If you are not familiar with how to install Nvidia driver and CUDA, please see below.

3. Build movienet-tools

    Get into movienet-tools folder, run
    
    ```python setup.py develop```
    
    or
    
    ```python -m pip install -e . -U```
    
    Note that if you want to install movienet-tools in a second anaconda environment,
    you should delete the ``build`` folder before running setup.

4. Dowload model weights

    ```python scripts/download_models.py```

5. Run a demo to see if you install it correctly

    ```python demos/face_demo.py```


### Example Install Nvidia driver 440 and CUDA 10.2
1. Remove exisitng nvidia
    ```sudo apt-get remove –purge nvidia*```

2. Blacklist nouveau
    ```sudo vim /etc/modprobe.d/blacklist.conf```
    Add `blacklist nouveau` at the end of the file
    
    ```sudo update-initramfs -u```
3. You may need to `reboot`

- You may need to stop the display `systemctl isolate multi-user.target` and `modprobe -r nvidia-drm`

4. Install Nvidia driver 440
    ```
    sudo chmod a+x NVIDIA-Linux-x86_64-xxxx.run
    sudo sh ./NVIDIA-Linux-x86_64-xxxx.run
    ```

5. Install CUDA 10.2
    ```sudo sh ./cuda_10.2.89_440.33.01_linux.run```
