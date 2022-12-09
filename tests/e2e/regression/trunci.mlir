// RUN: iree-run-mlir --iree-hal-target-backends=llvm-cpu %s --function_input=i32=-128 | FileCheck %s
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]]    || (iree-run-mlir --iree-hal-target-backends=vmvx %s --function_input=i32=-128 | FileCheck %s)

// CHECK: EXEC @trunci_i8
func.func @trunci_i8(%arg0: tensor<i32>) -> (tensor<f32>) {
  %0 = arith.trunci %arg0 : tensor<i32> to tensor<i8>
  %1 = arith.sitofp %0 : tensor<i8> to tensor<f32>
  // CHECK: f32=-128
  return %1 : tensor<f32>
}

