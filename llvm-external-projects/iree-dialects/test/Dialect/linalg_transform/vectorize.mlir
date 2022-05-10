// RUN: iree-dialects-opt -linalg-transform-interp -linalg-transform-file-name=%p/vectorize-transforms.mlir %s | FileCheck %s

// NOCHECK-LABEL: func @matmul_tensors(
// NOCHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// NOCHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// NOCHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// NOCHECK-SAME:  -> tensor<128x128xf32> {
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // NOCHECK: %[[VA:.*]] = vector.transfer_read %[[TA]]
  // NOCHECK: %[[VB:.*]] = vector.transfer_read %[[TB]]
  // NOCHECK: %[[VC:.*]] = vector.transfer_read %[[TC]]
  // NOCHECK: %[[VCU:.*]] = vector.contract {{.*}} %[[VA]], %[[VB]], %[[VC]]
  // NOCHECK: vector.transfer_write %[[VCU]], %[[TC]]
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}
