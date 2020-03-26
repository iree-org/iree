// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @slice_whole_buffer
func @slice_whole_buffer() -> tensor<3x4xi32> attributes { sym_visibility = "private" } {
  // CHECK: [[SRC:%.+]] = "vmla.constant"()
  %input = constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c48)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: [[SRC]], %rs3_4, %c0, %c0,
  // CHECK-SAME: [[DST]], %rs3_4, %c0, %c0,
  // CHECK-SAME: %c3, %c4
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
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c16)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: [[SRC]], %rs3_4, %c1, %c0,
  // CHECK-SAME: [[DST]], %rs1_4, %c0, %c0,
  // CHECK-SAME: %c1, %c4
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
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c8)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: [[SRC]], %rs3_4, %c1, %c1,
  // CHECK-SAME: [[DST]], %rs1_2, %c0, %c0,
  // CHECK-SAME: %c1, %c2
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
  // CHECK: [[DST:%.+]] = "vmla.buffer.alloc"(%c32)
  // CHECK: "vmla.copy"(
  // CHECK-SAME: [[SRC]], %rs3_4, %c1, %c0,
  // CHECK-SAME: [[DST]], %rs2_4, %c0, %c0,
  // CHECK-SAME: %c2, %c4
  // CHECK-SAME: ) {element_type = i32}
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<2x4xi32>
  // CHECK-NEXT: return [[DST]]
  return %result : tensor<2x4xi32>
}
