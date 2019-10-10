// RUN: iree-opt --lower-xla-to-iree-interpreter %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: @gather
// CHECK-SAME: [[INPUT:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[START_INDICES:%[a-zA-Z0-9]+]]
func @gather(%input : tensor<5x2x3xf32>, %start_indices : tensor<i64>) -> tensor<2x3xf32> {
  // CHECK-DAG:  [[SRC:%.+]] = iree.tensor_to_memref([[INPUT]] : tensor<5x2x3xf32>)
  // CHECK-DAG:  [[START_INDICES_MEMREF:%.+]] = iree.tensor_to_memref([[START_INDICES]] : tensor<i64>)
  // CHECK-DAG:  [[START_INDICES_NEW_SHAPE:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG:  [[START_INDICES_RESHAPED:%.+]] = "iree_hl_interp.reshape"([[START_INDICES_MEMREF]], [[START_INDICES_NEW_SHAPE]])
  // CHECK-DAG:  [[ZEROES:%.+]] = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-DAG:  [[START_INDICES_PADDED:%.+]] = "iree_hl_interp.concat"([[START_INDICES_RESHAPED]], [[ZEROES]])
  // CHECK-DAG:  [[DST:%.+]] = "iree_hl_interp.alloc_heap"() : () -> memref<1x2x3xf32>
  // CHECK-DAG:  [[DST_INDICES:%.+]] = iree.constant[dense<0>
  // CHECK-DAG:  [[LENGTHS:%.+]] = iree.constant[dense<[1, 2, 3]>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[START_INDICES_PADDED]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-DAG:  [[NEW_SHAPE:%.+]] = iree.constant[dense<[2, 3]>
  // CHECK-DAG:  [[RESHAPED:%.+]] = "iree_hl_interp.reshape"([[DST]], [[NEW_SHAPE]])
  // CHECK-DAG:  [[RESULT_TENSOR:%.+]] = iree.memref_to_tensor([[RESHAPED]] : memref<2x3xf32>)
  %result = "xla_hlo.gather"(%input, %start_indices) {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>,
      start_index_map = dense<0> : tensor<1xi64>
  } : (tensor<5x2x3xf32>, tensor<i64>) -> tensor<2x3xf32>
  // CHECK-NEXT: return [[RESULT_TENSOR]]
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: @gather_nonscalar_indices
// CHECK-SAME: [[INPUT:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[START_INDICES:%[a-zA-Z0-9]+]]
func @gather_nonscalar_indices(%input : tensor<5x2x3xf32>, %start_indices : tensor<1xi64>) -> tensor<2x3xf32> {
  // CHECK-DAG:  [[SRC:%.+]] = iree.tensor_to_memref([[INPUT]] : tensor<5x2x3xf32>)
  // CHECK-DAG:  [[START_INDICES_MEMREF:%.+]] = iree.tensor_to_memref([[START_INDICES]] : tensor<1xi64>)
  // CHECK-DAG:  [[ZEROES:%.+]] = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-DAG:  [[START_INDICES_PADDED:%.+]] = "iree_hl_interp.concat"([[START_INDICES_MEMREF]], [[ZEROES]])
  // CHECK-DAG:  [[DST:%.+]] = "iree_hl_interp.alloc_heap"() : () -> memref<1x2x3xf32>
  // CHECK-DAG:  [[DST_INDICES:%.+]] = iree.constant[dense<0>
  // CHECK-DAG:  [[LENGTHS:%.+]] = iree.constant[dense<[1, 2, 3]>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[START_INDICES_PADDED]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-DAG:  [[NEW_SHAPE:%.+]] = iree.constant[dense<[2, 3]>
  // CHECK-DAG:  [[RESHAPED:%.+]] = "iree_hl_interp.reshape"([[DST]], [[NEW_SHAPE]])
  // CHECK-DAG:  [[RESULT_TENSOR:%.+]] = iree.memref_to_tensor([[RESHAPED]] : memref<2x3xf32>)
  %result = "xla_hlo.gather"(%input, %start_indices) {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>,
      start_index_map = dense<0> : tensor<1xi64>
  } : (tensor<5x2x3xf32>, tensor<1xi64>) -> tensor<2x3xf32>
  // CHECK-NEXT: return [[RESULT_TENSOR]]
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: @gather_fully_specified_indices
// CHECK-SAME: [[INPUT:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[START_INDICES:%[a-zA-Z0-9]+]]
func @gather_fully_specified_indices(%input : tensor<5x2x3xf32>, %start_indices : tensor<3xi64>) -> tensor<2x3xf32> {
  // CHECK-DAG:  [[SRC:%.+]] = iree.tensor_to_memref([[INPUT]] : tensor<5x2x3xf32>)
  // CHECK-DAG:  [[START_INDICES_MEMREF:%.+]] = iree.tensor_to_memref([[START_INDICES]] : tensor<3xi64>)
  // CHECK-DAG:  [[DST:%.+]] = "iree_hl_interp.alloc_heap"() : () -> memref<1x2x3xf32>
  // CHECK-DAG:  [[DST_INDICES:%.+]] = iree.constant[dense<0>
  // CHECK-DAG:  [[LENGTHS:%.+]] = iree.constant[dense<[1, 2, 3]>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[START_INDICES_MEMREF]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-DAG:  [[NEW_SHAPE:%.+]] = iree.constant[dense<[2, 3]>
  // CHECK-DAG:  [[RESHAPED:%.+]] = "iree_hl_interp.reshape"([[DST]], [[NEW_SHAPE]])
  // CHECK-DAG:  [[RESULT_TENSOR:%.+]] = iree.memref_to_tensor([[RESHAPED]] : memref<2x3xf32>)
  %result = "xla_hlo.gather"(%input, %start_indices) {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>,
      start_index_map = dense<0> : tensor<1xi64>
  } : (tensor<5x2x3xf32>, tensor<3xi64>) -> tensor<2x3xf32>
  // CHECK-NEXT: return [[RESULT_TENSOR]]
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: @gather_not_lowered
// CHECK-SAME: [[INPUT:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[START_INDICES:%[a-zA-Z0-9]+]]
func @gather_not_lowered(%input : tensor<5x2x3xf32>, %start_indices : tensor<2x2xi64>) {
  // CHECK-NEXT "xla_hlo.gather"
  %axis_1 = "xla_hlo.gather"(%input, %start_indices) {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 1 : i64,
      offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
      slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>,
      start_index_map = dense<0> : tensor<1xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>

  // CHECK-NEXT "xla_hlo.gather"
  %collapse_1 = "xla_hlo.gather"(%input, %start_indices) {
      collapsed_slice_dims = dense<1> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
      slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>,
      start_index_map = dense<0> : tensor<1xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>

  // CHECK-NEXT "xla_hlo.gather"
  %transposes = "xla_hlo.gather"(%input, %start_indices) {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
      slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>,
      start_index_map = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>

  // CHECK-NEXT "xla_hlo.gather"
  %has_batch_dims = "xla_hlo.gather"(%input, %start_indices) {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<1> : tensor<1xi64>,
      slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>,
      start_index_map = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>
  return
}
