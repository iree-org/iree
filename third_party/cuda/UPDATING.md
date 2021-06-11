Those headers come from CUDA SDK.

To update, install CUDA SDK locally:
```
sudo apt-get install cuda
```

Copy cuda.h, version.txt and libdevice.10.bc:
```
cp /usr/local/cuda/include/cuda.h ./include/
cp /usr/local/cuda/version.txt  .
cp /usr/local/cuda/nvvm/libdevice/libdevice.10.bc ./nvvm/libdevice/
```
