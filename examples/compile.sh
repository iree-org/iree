#!/bin/bash
#
# cat ../resnet18_stablehlo.mlir | 
# iree-opt --iree-mhlo-to-linalg-on-tensors 2>&1 |tee resnet18_linalg.mlir 
# 
# iree-compile \
#     --iree-hal-target-backends=cuda \
#     --iree-hal-cuda-llvm-target-arch=sm_60 \
#     --iree-codegen
#     resnet18_linalg.mlir -o resnet_cuda.vmfb


#===------------------===#
# cuda
#===------------------===#
#iree-opt --iree-mhlo-input-transformation-pipeline  \
#
iree-opt --iree-stablehlo-legalize-chlo --iree-stablehlo-to-iree-input \
        --split-input-file examples/tinymodel_stablehlo.mlir 2>&1 |tee elementwise-add-linalg.mlir
iree-compile --iree-hal-target-backends=cuda \
        elementwise-add-linalg.mlir -o elementwiseadd.vmfb --mlir-print-ir-after-all 2>&1 |tee log-elementwise-add.mlir
iree-compile --iree-hal-target-backends=cuda elementwise-add-linalg.mlir \
        -o elementwise-add.vmfb \
            --print-after-all 2>&1 |tee log-elementwise-add.ll
iree-run-module   --device=cuda   --module=elementwiseadd.vmfb   --function=forward --input="1x4xf32=[1,1,1,1]"



#===------------------===#
# vulkan 
#===------------------===#
iree-opt --iree-stablehlo-legalize-chlo --iree-stablehlo-to-iree-input \
        --split-input-file examples/tinymodel_stablehlo.mlir 2>&1 |tee elementwise-add-linalg.mlir
iree-compile --iree-hal-target-backends=vulkan \
        elementwise-add-linalg.mlir -o elementwiseadd.vmfb --mlir-print-ir-after-all 2>&1 |tee log-elementwise-add.mlir
iree-compile --iree-hal-target-backends=vulkan elementwise-add-linalg.mlir \
        -o elementwise-add.vmfb \
            --print-after-all 2>&1 |tee log-elementwise-add.ll
iree-run-module   --device=vulkan   --module=elementwiseadd.vmfb   --function=forward --input="1x4xf32=[1,1,1,1]"
