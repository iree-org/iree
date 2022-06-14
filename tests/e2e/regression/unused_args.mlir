// RUN: iree-run-mlir --iree-hal-target-backends=vmvx --function-input=4xf32=0,0,0,0 --function-input=4xf32=1,1,1,1 %s | FileCheck %s
// RUN: iree-run-mlir --iree-hal-target-backends=dylib-llvm-aot --function-input=4xf32=0,0,0,0 --function-input=4xf32=1,1,1,1 %s | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --iree-hal-target-backends=vulkan-spirv --function-input=4xf32=0,0,0,0 --function-input=4xf32=1,1,1,1 %s | FileCheck %s)

// CHECK-LABEL: EXEC @arg0_unused
func.func @arg0_unused(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  return %arg1 : tensor<4xf32>
}
// CHECK: 4xf32=1 1 1 1
