// RUN: iree-run-mlir -iree-hal-target-backends=vmla -input-value=4xf32=0,0,0,0 -input-value=4xf32=1,1,1,1 %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=llvm-ir -input-value=4xf32=0,0,0,0 -input-value=4xf32=1,1,1,1 %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -input-value=4xf32=0,0,0,0 -input-value=4xf32=1,1,1,1 %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @arg0_unused
func @arg0_unused(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {iree.module.export} {
  return %arg1 : tensor<4xf32>
}
// CHECK: 4xf32=1 1 1 1
