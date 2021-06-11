// RUN: iree-run-mlir -export-all -iree-hal-target-backends=vmvx -function-input=4xf32=0,0,0,0 -function-input=4xf32=1,1,1,1 %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=dylib-llvm-aot -function-input=4xf32=0,0,0,0 -function-input=4xf32=1,1,1,1 %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=vulkan-spirv -function-input=4xf32=0,0,0,0 -function-input=4xf32=1,1,1,1 %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @arg0_unused
func @arg0_unused(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {iree.module.export} {
  return %arg1 : tensor<4xf32>
}
// CHECK: 4xf32=1 1 1 1
