// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-materialize-calculations %s | IreeFileCheck %s

// CHECK-LABEL: @transpose
// CHECK-SAME: %[[ARG:[^:[:space:]]+]]: tensor<?x7x10xf32>
func @transpose(%arg0 : tensor<?x7x10xf32>) -> (tensor<7x?x10xf32>, !shapex.ranked_shape<[7,?,10]>) {
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} :
      (tensor<?x7x10xf32>) -> tensor<7x?x10xf32>
  // CHECK-DAG: %[[RESULT:.+]] = "xla_hlo.transpose"
  // Note that the dim is dependent on the arg due to fallback runtime resolution.
  // The key is that the transpose shape function mapped the dimension
  // properly.
  // CHECK-DAG: %[[DIM:.+]] = dim %[[ARG]], 0
  // CHECK-DAG: %[[SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
  %1 = shapex.get_ranked_shape %0 : tensor<7x?x10xf32> -> !shapex.ranked_shape<[7,?,10]>
  // CHECK: return %[[RESULT]], %[[SHAPE]]
  return %0, %1 : tensor<7x?x10xf32>, !shapex.ranked_shape<[7,?,10]>
}
