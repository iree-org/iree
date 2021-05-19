Those headers come from ROCM SDK.

Currently updates are not supported by ROCm, so we need to uninstall and reinstall ROCm if we want to update
To update, install ROCM SDK locally:
```
sudo apt autoremove rocm-opencl rocm-dkms rocm-dev rocm-utils && sudo reboot
sudo apt-get install rocm-dkms
```

Copy HIP and HSA headers, version.txt and libdevice.10.bc:
```
cp -RL /opt/rocm/include/hip ./include/
cp -RL /opt/rocm/include/hsa ./include/
cp /opt/rocm/.info/version version.txt
```
