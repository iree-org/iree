// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir %s -iree-hal-target-backends=dylib-llvm-aot -function-input="2x3xi64" | IreeFileCheck %s)
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]]    || (iree-run-mlir %s -iree-hal-target-backends=vmvx           -function-input="2x3xi64" | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]]  || (iree-run-mlir %s -iree-hal-target-backends=vulkan-spirv   -function-input="2x3xi64" | IreeFileCheck %s)

// CHECK: EXEC @fill_i64
func @fill_i64(%arg0: tensor<?x?xi64>) -> (tensor<?x?xi64>, tensor<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xi64>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xi64>

  %cv0 = arith.constant -1 : i64
  %v0_init = linalg.init_tensor [%0, %1] : tensor<?x?xi64>
  %v0 = linalg.fill(%cv0, %v0_init) : i64, tensor<?x?xi64> -> tensor<?x?xi64>
  // CHECK: 2x3xi64=[-1 -1 -1][-1 -1 -1]

  %cv1 = arith.constant 9223372036854775807 : i64
  %v1_init = linalg.init_tensor [%0, %1] : tensor<?x?xi64>
  %v1 = linalg.fill(%cv1, %v1_init) : i64, tensor<?x?xi64> -> tensor<?x?xi64>
  // CHECK: 2x3xi64=[9223372036854775807 9223372036854775807 9223372036854775807][9223372036854775807 9223372036854775807 9223372036854775807]

  return %v0, %v1 : tensor<?x?xi64>, tensor<?x?xi64>
}
