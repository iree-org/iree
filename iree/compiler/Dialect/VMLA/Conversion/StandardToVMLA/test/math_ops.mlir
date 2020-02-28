// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @absf
func @absf(%arg0 : tensor<4xf32>) -> tensor<4xf32> attributes { sym_visibility = "private" } {
  // CHECK-NEXT: [[BUF_SZ:%.+]] = constant 16
  // CHECK-NEXT: [[BUF:%.+]] = "vmla.buffer.alloc"([[BUF_SZ]])
  // CHECK-NEXT: "vmla.abs"(%arg0, [[BUF]]) {element_type = f32}
  %0 = absf %arg0 : tensor<4xf32>
  // CHECK-NEXT: return [[BUF]]
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @shr_signed
func @shr_signed(%arg0 : tensor<4xi32>) -> tensor<4xi32> attributes { sym_visibility = "private" } {
  // CHECK-NEXT: [[BUF_SZ:%.+]] = constant 16
  // CHECK-NEXT: [[BUF:%.+]] = "vmla.buffer.alloc"([[BUF_SZ]])
  // CHECK-NEXT: "vmla.shr"(%arg0, %arg0, [[BUF]]) {element_type = i32}
  %0 = shift_right_signed %arg0, %arg0 : tensor<4xi32>
  // CHECK-NEXT: return [[BUF]]
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @shr_unsigned
func @shr_unsigned(%arg0 : tensor<4xi32>) -> tensor<4xi32> attributes { sym_visibility = "private" } {
  // CHECK-NEXT: [[BUF_SZ:%.+]] = constant 16
  // CHECK-NEXT: [[BUF:%.+]] = "vmla.buffer.alloc"([[BUF_SZ]])
  // CHECK-NEXT: "vmla.shr"(%arg0, %arg0, [[BUF]]) {element_type = i32, force_unsigned}
  %0 = shift_right_unsigned %arg0, %arg0 : tensor<4xi32>
  // CHECK-NEXT: return [[BUF]]
  return %0 : tensor<4xi32>
}
