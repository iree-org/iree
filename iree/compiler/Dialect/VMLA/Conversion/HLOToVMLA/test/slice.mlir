// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @slice_whole_stride
func @slice_whole_stride(%arg0 : tensor<3x4xi32>) -> tensor<1x4xi32> attributes { sym_visibility = "private" } {
  // CHECK-SAME: %[[SRC:.+]]:
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c16
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: src_indices = [%c1, %c0],
  // CHECK-SAME: out %[[DST]](%rs1_4 : !shapex.ranked_shape<[1,4]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c1, %c4] : i32
  %result = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x4xi32>
  // CHECK-NEXT: return %[[DST]]
  return %result : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @slice_stride_part
func @slice_stride_part(%arg0 : tensor<3x4xi32>) -> tensor<1x2xi32> attributes { sym_visibility = "private" } {
  // CHECK-SAME: %[[SRC:.+]]:
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c8
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: src_indices = [%c1, %c1],
  // CHECK-SAME: out %[[DST]](%rs1_2 : !shapex.ranked_shape<[1,2]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c1, %c2] : i32
  %result = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 1]> : tensor<2xi64>,
    limit_indices = dense<[2, 3]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  // CHECK-NEXT: return %[[DST]]
  return %result : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: @slice_multi_stride
func @slice_multi_stride(%arg0: tensor<3x4xi32>) -> tensor<2x4xi32> attributes { sym_visibility = "private" } {
  // CHECK-SAME: %[[SRC:.+]]:
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c32
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%rs3_4 : !shapex.ranked_shape<[3,4]>),
  // CHECK-SAME: src_indices = [%c1, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_4 : !shapex.ranked_shape<[2,4]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c2, %c4] : i32
  %result = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<2x4xi32>
  // CHECK-NEXT: return %[[DST]]
  return %result : tensor<2x4xi32>
}
