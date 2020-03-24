// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s -verify-diagnostics | IreeFileCheck %s

// CHECK-LABEL: @gather_scalar_indices
// CHECK-SAME: [[SRC:%.+]]: !vmla.buffer,
// CHECK-SAME: [[INDICES:%.+]]: !vmla.buffer)
func @gather_scalar_indices(%input : tensor<5x1x5xi32>, %start_indices : tensor<i64>) -> tensor<1x5xi32> attributes { sym_visibility = "private" } {
  // CHECK-DAG: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[5,1,5]>
  // CHECK-DAG: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[1,1,5]>
  // CHECK-DAG: [[INDEX0:%.+]] = "vmla.buffer.load.i32"([[INDICES]], %c0_i32)
  // CHECK-DAG: [[DST:%.+]] = "vmla.buffer.alloc"(%c20_i32)
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[SRC]], [[SRC_SHAPE]], [[INDEX0]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32, %c0_i32,
  // CHECK-SAME: %c1_i32, %c1_i32, %c5_i32
  // CHECK-SAME: ) {element_type = i32}
  %0 = "xla_hlo.gather"(%input, %start_indices) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      start_index_map = dense<0> : tensor<1xi64>
    },
    slice_sizes = dense<[1, 1, 5]> : tensor<3xi64>
  } : (tensor<5x1x5xi32>, tensor<i64>) -> tensor<1x5xi32>
  // CHECK-NEXT: return [[DST]]
  return %0 : tensor<1x5xi32>
}

// -----

// CHECK-LABEL: @gather_fully_specified_indices
// CHECK-SAME: [[SRC:%.+]]: !vmla.buffer,
// CHECK-SAME: [[INDICES:%.+]]: !vmla.buffer)
func @gather_fully_specified_indices(%input : tensor<5x2x3xf32>, %start_indices : tensor<3xi64>) -> tensor<2x3xf32> attributes { sym_visibility = "private" } {
  // CHECK-DAG: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[5,2,3]>
  // CHECK-DAG: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[1,2,3]>
  // CHECK-DAG: [[INDEX0:%.+]] = "vmla.buffer.load.i32"([[INDICES]], %c0_i32)
  // CHECK-DAG: [[INDEX1:%.+]] = "vmla.buffer.load.i32"([[INDICES]], %c4_i32)
  // CHECK-DAG: [[INDEX2:%.+]] = "vmla.buffer.load.i32"([[INDICES]], %c8_i32)
  // CHECK-DAG: [[DST:%.+]] = "vmla.buffer.alloc"(%c24_i32)
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[SRC]], [[SRC_SHAPE]], [[INDEX0]], [[INDEX1]], [[INDEX2]],
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32, %c0_i32,
  // CHECK-SAME: %c1_i32, %c2_i32, %c3_i32
  // CHECK-SAME: ) {element_type = f32}
  %0 = "xla_hlo.gather"(%input, %start_indices) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      start_index_map = dense<0> : tensor<1xi64>
    },
    slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>
  } : (tensor<5x2x3xf32>, tensor<3xi64>) -> tensor<2x3xf32>
  // CHECK-NEXT: return [[DST]]
  return %0 : tensor<2x3xf32>
}

// -----

// expected-error@-3 {{conversion to the VMLA dialect failed}}
func @gather_not_lowered_axis_1(%input : tensor<5x2x3xf32>, %start_indices : tensor<2x2xi64>) attributes { sym_visibility = "private" } {
  // expected-remark@+2 {{couldn't lower gather}}
  // expected-error@+1 {{failed to legalize operation 'xla_hlo.gather' that was explicitly marked illegal}}
  %0 = "xla_hlo.gather"(%input, %start_indices) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 1 : i64,
      offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
      start_index_map = dense<0> : tensor<1xi64>
    },
    slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>
  return
}

// -----

// expected-error@-3 {{conversion to the VMLA dialect failed}}
func @gather_not_lowered_collapse(%input : tensor<5x2x3xf32>, %start_indices : tensor<2x2xi64>) attributes { sym_visibility = "private" } {
  // expected-remark@+2 {{couldn't lower gather}}
  // expected-error@+1 {{failed to legalize operation 'xla_hlo.gather' that was explicitly marked illegal}}
  %0 = "xla_hlo.gather"(%input, %start_indices) {
    dimension_numbers = {
      collapsed_slice_dims = dense<1> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
      start_index_map = dense<0> : tensor<1xi64>
    },
    slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>
  return
}

// -----

// expected-error@-3 {{conversion to the VMLA dialect failed}}
func @gather_not_lowered_transposes(%input : tensor<5x2x3xf32>, %start_indices : tensor<2x2xi64>) attributes { sym_visibility = "private" } {
  // expected-remark@+2 {{couldn't lower gather}}
  // expected-error@+1 {{failed to legalize operation 'xla_hlo.gather' that was explicitly marked illegal}}
  %0 = "xla_hlo.gather"(%input, %start_indices) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
      start_index_map = dense<[1, 0]> : tensor<2xi64>
    },
    slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>
  return
}

// -----

// expected-error@-3 {{conversion to the VMLA dialect failed}}
func @gather_not_lowered_batch_dims(%input : tensor<5x2x3xf32>, %start_indices : tensor<2x2xi64>) attributes { sym_visibility = "private" } {
  // expected-remark@+2 {{couldn't lower gather}}
  // expected-error@+1 {{failed to legalize operation 'xla_hlo.gather' that was explicitly marked illegal}}
  %0 = "xla_hlo.gather"(%input, %start_indices) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<1> : tensor<1xi64>,
      start_index_map = dense<[1, 0]> : tensor<2xi64>
    },
    slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>
  } : (tensor<5x2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>
  return
}
