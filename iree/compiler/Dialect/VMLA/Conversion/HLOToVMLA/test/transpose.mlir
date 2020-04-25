// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @transpose
func @transpose() -> tensor<24x7x10xf32> attributes { sym_visibility = "private" } {
  // CHECK-DAG: %[[SRC:.+]] = "vmla.constant"()
  %input = constant dense<1.0> : tensor<7x24x10xf32>
  // CHECK-DAG: %[[SRC_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[7,24,10]>
  // CHECK-DAG: %[[DST_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[24,7,10]>
  // CHECK-DAG: %[[DST_SIZE:.+]] = constant 6720 : index
  // CHECK-DAG: %[[DST:.+]] = "vmla.buffer.alloc"(%[[DST_SIZE]])
  // CHECK-NEXT: "vmla.transpose"(
  // CHECK-SAME:   %[[SRC]], %[[SRC_SHAPE]],
  // CHECK-SAME:   %[[DST]], %[[DST_SHAPE]]
  // CHECK-SAME: ) {element_type = f32, permutation = dense<[1, 0, 2]> : tensor<3xi32>}
  %0 = "xla_hlo.transpose"(%input) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<7x24x10xf32>) -> tensor<24x7x10xf32>
  // CHECK-NEXT: return %[[DST]]
  return %0 : tensor<24x7x10xf32>
}
