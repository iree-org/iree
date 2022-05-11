// RUN: iree-dialects-opt --linalg-transform-interp --linalg-transform-file-name=%p/vectorize-transforms.mlir %s | FileCheck %s

// CHECK-LABEL: func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: %[[VA:.*]] = vector.transfer_read %[[TA]]
  // CHECK: %[[VB:.*]] = vector.transfer_read %[[TB]]
  // CHECK: %[[VC:.*]] = vector.transfer_read %[[TC]]
  // CHECK: %[[VCU:.*]] = vector.contract {{.*}} %[[VA]], %[[VB]], %[[VC]]
  // CHECK: vector.transfer_write %[[VCU]], %[[TC]]
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}
