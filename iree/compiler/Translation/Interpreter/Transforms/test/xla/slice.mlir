// RUN: iree-opt --lower-xla-to-iree-interpreter %s | IreeFileCheck %s

// CHECK-LABEL: @slice
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @slice(%arg : tensor<3x4xf32>) -> tensor<1x4xf32> {
  // CHECK-DAG:  [[SRC:%.+]]   = iree_interp.tensor_to_memref([[ARG]]
  // CHECK-DAG:  [[SRC_INDICES:%.+]] = iree_interp.constant[dense<[1, 0]>
  // CHECK-DAG:  [[DST:%.+]]     = "iree_hl_interp.alloc_heap"() : () -> memref<1x4xf32>
  // CHECK-DAG:  [[DST_INDICES:%.+]] = iree_interp.constant[dense<0>
  // CHECK-DAG:  [[LENGTHS:%.+]] = iree_interp.constant[dense<[1, 4]>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: [[RESULT_TENSOR:%.+]] = iree_interp.memref_to_tensor([[DST]]
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: return [[RESULT_TENSOR]]
  return %result : tensor<1x4xf32>
}

// CHECK-LABEL: @slice_noncontiguous
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @slice_noncontiguous(%arg : tensor<3x4xf32>) -> tensor<2x2xf32> {
  // CHECK-DAG:  [[SRC:%.+]]   = iree_interp.tensor_to_memref([[ARG]]
  // CHECK-DAG:  [[SRC_INDICES:%.+]] = iree_interp.constant[dense<1>
  // CHECK-DAG:  [[DST:%.+]]     = "iree_hl_interp.alloc_heap"() : () -> memref<2x2xf32>
  // CHECK-DAG:  [[DST_INDICES:%.+]] = iree_interp.constant[dense<0>
  // CHECK-DAG:  [[LENGTHS:%.+]] = iree_interp.constant[dense<2>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: [[RESULT_TENSOR:%.+]] = iree_interp.memref_to_tensor([[DST]]
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<1> : tensor<2xi64>, limit_indices = dense<3> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: return [[RESULT_TENSOR]]
  return %result : tensor<2x2xf32>
}
