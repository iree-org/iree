#!/bin/bash
#
cat ../resnet18_stablehlo.mlir | 
iree-opt --iree-mhlo-to-linalg-on-tensors 2>&1 |tee resnet18_linalg.mlir 

iree-compile \
    --iree-hal-target-backends=cuda \
    --iree-hal-cuda-llvm-target-arch=sm_60 \
    --iree-codegen
    resnet18_linalg.mlir -o resnet_cuda.vmfb

