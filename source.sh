#!/bin/bash
#
export TORCHMLIR=/mnt/compiler_workspace/torch-mlir-lj
export PYTHONPATH=$TORCHMLIR/build/tools/torch-mlir/python_packages/torch_mlir:$TORCHMLIR/examples:$PYTHONPATH

export IREE=/mnt/compiler_workspace/iree/
source $IREE/iree-build-host/.env && export PYTHONPATH
export PATH=$PWD/iree-build-host/tools/:$PATH
