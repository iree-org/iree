// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @slice_whole_buffer
func @slice_whole_buffer() -> tensor<3x4xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[SRC:%.+]] = "vmla.constant"()
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-NEXT: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST:%.+]] = "vmla.buffer.alloc"(%c48_i32)
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[SRC]], [[SRC_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c3_i32, %c4_i32
  // CHECK-SAME: ) {element_type = i32}
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[0, 0]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<3x4xi32>
  // CHECK-NEXT: return [[DST]]
  return %result : tensor<3x4xi32>
}

// -----

// CHECK-LABEL: @slice_whole_stride
func @slice_whole_stride() -> tensor<1x4xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[SRC:%.+]] = "vmla.constant"()
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-NEXT: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST:%.+]] = "vmla.buffer.alloc"(%c16_i32)
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[SRC]], [[SRC_SHAPE]], %c1_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c1_i32, %c4_i32
  // CHECK-SAME: ) {element_type = i32}
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x4xi32>
  // CHECK-NEXT: return [[DST]]
  return %result : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @slice_stride_part
func @slice_stride_part() -> tensor<1x2xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[SRC:%.+]] = "vmla.constant"()
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-NEXT: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST:%.+]] = "vmla.buffer.alloc"(%c8_i32)
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[SRC]], [[SRC_SHAPE]], %c1_i32, %c1_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c1_i32, %c2_i32
  // CHECK-SAME: ) {element_type = i32}
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[1, 1]> : tensor<2xi64>,
    limit_indices = dense<[2, 3]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  // CHECK-NEXT: return [[DST]]
  return %result : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: @slice_multi_stride
func @slice_multi_stride() -> tensor<2x4xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[SRC:%.+]] = "vmla.constant"()
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK-NEXT: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape
  // CHECK-NEXT: [[DST:%.+]] = "vmla.buffer.alloc"(%c32_i32)
  // CHECK-NEXT: "vmla.copy"(
  // CHECK-SAME: [[SRC]], [[SRC_SHAPE]], %c1_i32, %c0_i32,
  // CHECK-SAME: [[DST]], [[DST_SHAPE]], %c0_i32, %c0_i32,
  // CHECK-SAME: %c2_i32, %c4_i32
  // CHECK-SAME: ) {element_type = i32}
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<2x4xi32>
  // CHECK-NEXT: return [[DST]]
  return %result : tensor<2x4xi32>
}
