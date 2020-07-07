// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @slice_whole_buffer
// CHECK-SAME: %[[SRC_IDX_1:.+]]: !vmla.buffer, %[[SRC_IDX_2:.+]]: !vmla.buffer
func @slice_whole_buffer(%src_idx_1 : tensor<i64>, %src_idx_2 : tensor<i64>) -> tensor<3x4xi32> attributes { sym_visibility = "private" } {
  // CHECK: %[[SRC:.+]] = vmla.constant
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-DAG: %[[SRC_INDEX_0_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_1]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_0:.+]] = index_cast %[[SRC_INDEX_0_I32]]
  // CHECK-DAG: %[[SRC_INDEX_1_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_2]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_1:.+]] = index_cast %[[SRC_INDEX_1_I32]]
  // CHECK-DAG: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c48 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: src_indices = [%[[SRC_INDEX_0]], %[[SRC_INDEX_1]]],
  // CHECK-SAME: out %[[DST]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c3, %c4] : i32
  %result = "mhlo.dynamic-slice"(%input, %src_idx_1, %src_idx_2) {
    slice_sizes = dense<[3, 4]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<3x4xi32>
  // CHECK-NEXT: return %[[DST]]
  return %result : tensor<3x4xi32>
}

// -----

// CHECK-LABEL: @slice_whole_stride
// CHECK-SAME: %[[SRC_IDX_1:.+]]: !vmla.buffer, %[[SRC_IDX_2:.+]]: !vmla.buffer
func @slice_whole_stride(%src_idx_1 : tensor<i64>, %src_idx_2 : tensor<i64>) -> tensor<1x4xi32> attributes { sym_visibility = "private" } {
  // CHECK: %[[SRC:.+]] = vmla.constant
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-DAG: %[[SRC_INDEX_0_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_1]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_0:.+]] = index_cast %[[SRC_INDEX_0_I32]]
  // CHECK-DAG: %[[SRC_INDEX_1_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_2]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_1:.+]] = index_cast %[[SRC_INDEX_1_I32]]
  // CHECK-DAG: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c16 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: src_indices = [%[[SRC_INDEX_0]], %[[SRC_INDEX_1]]],
  // CHECK-SAME: out %[[DST]](%rs1_4 : !shapex.ranked_shape<[1,4]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c1, %c4] : i32
  %result = "mhlo.dynamic-slice"(%input, %src_idx_1, %src_idx_2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK-NEXT: return %[[DST]]
  return %result : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @slice_stride_part
// CHECK-SAME: %[[SRC_IDX_1:.+]]: !vmla.buffer, %[[SRC_IDX_2:.+]]: !vmla.buffer
func @slice_stride_part(%src_idx_1 : tensor<i64>, %src_idx_2 : tensor<i64>) -> tensor<1x2xi32> attributes { sym_visibility = "private" } {
  // CHECK: %[[SRC:.+]] = vmla.constant
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-DAG: %[[SRC_INDEX_0_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_1]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_0:.+]] = index_cast %[[SRC_INDEX_0_I32]]
  // CHECK-DAG: %[[SRC_INDEX_1_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_2]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_1:.+]] = index_cast %[[SRC_INDEX_1_I32]]
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c8 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: src_indices = [%[[SRC_INDEX_0]], %[[SRC_INDEX_1]]],
  // CHECK-SAME: out %[[DST]](%rs1_2 : !shapex.ranked_shape<[1,2]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c1, %c2] : i32
  %result = "mhlo.dynamic-slice"(%input, %src_idx_1, %src_idx_2) {
    slice_sizes = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x2xi32>
  // CHECK-NEXT: return %[[DST]]
  return %result : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: @slice_multi_stride
// CHECK-SAME: %[[SRC_IDX_1:.+]]: !vmla.buffer, %[[SRC_IDX_2:.+]]: !vmla.buffer
func @slice_multi_stride(%src_idx_1 : tensor<i64>, %src_idx_2 : tensor<i64>) -> tensor<2x4xi32> attributes { sym_visibility = "private" } {
  // CHECK: %[[SRC:.+]] = vmla.constant
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-DAG: %[[SRC_INDEX_0_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_1]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_0:.+]] = index_cast %[[SRC_INDEX_0_I32]]
  // CHECK-DAG: %[[SRC_INDEX_1_I32:.+]] = vmla.buffer.load.i32 %[[SRC_IDX_2]][%c0] : i32
  // CHECK-DAG: %[[SRC_INDEX_1:.+]] = index_cast %[[SRC_INDEX_1_I32]]
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c32 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: src_indices = [%[[SRC_INDEX_0]], %[[SRC_INDEX_1]]],
  // CHECK-SAME: out %[[DST]](%rs2_4 : !shapex.ranked_shape<[2,4]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c2, %c4] : i32
  %result = "mhlo.dynamic-slice"(%input, %src_idx_1, %src_idx_2) {
    slice_sizes = dense<[2, 4]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x4xi32>
  // CHECK-NEXT: return %[[DST]]
  return %result : tensor<2x4xi32>
}
