// RUN: iree-dialects-opt --transform-dialect-interpreter %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: memref<128x128xf32
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: memref<128x128xf32
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: memref<128x128xf32
// CHECK-NOT:   -> tensor
func.func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: linalg.matmul ins(%[[TA]], %[[TB]] : memref{{.*}}, memref{{.*}} outs(%[[TC]] : memref{{.*}})
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  // CHECK: return %[[TC]]
  return %0 : tensor<128x128xf32>
// CHECK: }
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  bufferize
}
