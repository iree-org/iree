// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode -input-value="2x2xf32= 1 2 3 4" -input-value="2x3xf32= 5 6 7 8 9 10" -input-value="2x2xf32= 11 12 13 14" %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -input-value="2x2xf32= 1 2 3 4" -input-value="2x3xf32= 5 6 7 8 9 10" -input-value="2x2xf32= 11 12 13 14" %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @xla_concatenate
func @xla_concatenate(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x2xf32>) -> (tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x7xf32>, tensor<4x2xf32>) {
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 1} : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
  %1 = "xla_hlo.concatenate"(%arg1, %arg0) {dimension = 1} : (tensor<2x3xf32>, tensor<2x2xf32>) -> tensor<2x5xf32>
  %2 = "xla_hlo.concatenate"(%arg0, %arg1, %arg2) {dimension = 1} : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x2xf32>) -> tensor<2x7xf32>
  %3 = "xla_hlo.concatenate"(%arg0, %arg2) {dimension = 0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  return %0, %1, %2, %3: tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x7xf32>, tensor<4x2xf32>
}
// CHECK: 2x5xf32=[1 2 5 6 7][3 4 8 9 10]
// CHECK-NEXT: 2x5xf32=[5 6 7 1 2][8 9 10 3 4]
// CHECK-NEXT: 2x7xf32=[1 2 5 6 7 11 12][3 4 8 9 10 13 14]
// CHECK-NEXT: 4x2xf32=[1 2][3 4][11 12][13 14]
