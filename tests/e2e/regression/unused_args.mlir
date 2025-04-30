// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx %s --input=4xf32=0 --input=4xf32=1 | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu %s --input=4xf32=0 --input=4xf32=1 | FileCheck %s

// CHECK-LABEL: EXEC @arg0_unused
func.func @arg0_unused(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  return %arg1 : tensor<4xf32>
}
// CHECK: 4xf32=1 1 1 1
