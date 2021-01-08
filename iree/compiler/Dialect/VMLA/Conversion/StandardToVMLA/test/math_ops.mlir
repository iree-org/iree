// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @absf
func private @absf(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[BUF_SZ:.+]] = constant 16
  // CHECK-NEXT: %[[BUF:.+]] = vmla.buffer.alloc byte_length = %[[BUF_SZ]] : !vmla.buffer
  // CHECK-NEXT: vmla.abs %arg0, out %[[BUF]] : f32
  %0 = absf %arg0 : tensor<4xf32>
  // CHECK-NEXT: return %[[BUF]]
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @shr_signed
func private @shr_signed(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT: %[[BUF_SZ:.+]] = constant 16
  // CHECK-NEXT: %[[BUF:.+]] = vmla.buffer.alloc byte_length = %[[BUF_SZ]] : !vmla.buffer
  // CHECK-NEXT: vmla.shr %arg0, %arg0, out %[[BUF]] : i32
  %0 = shift_right_signed %arg0, %arg0 : tensor<4xi32>
  // CHECK-NEXT: return %[[BUF]]
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @shr_unsigned
func private @shr_unsigned(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT: %[[BUF_SZ:.+]] = constant 16
  // CHECK-NEXT: %[[BUF:.+]] = vmla.buffer.alloc byte_length = %[[BUF_SZ]] : !vmla.buffer
  // CHECK-NEXT: vmla.shr %arg0, %arg0, out %[[BUF]] {force_unsigned} : i32
  %0 = shift_right_unsigned %arg0, %arg0 : tensor<4xi32>
  // CHECK-NEXT: return %[[BUF]]
  return %0 : tensor<4xi32>
}
