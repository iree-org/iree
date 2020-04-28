// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @cmp_i
func @cmp_i(%arg0 : tensor<4xi32>, %arg1 : tensor<4xi32>) -> tensor<4xi1> attributes { sym_visibility = "private" } {
  // CHECK: %[[BUF_SZ:.+]] = constant 4
  // CHECK-NEXT: %[[BUF:.+]] = vmla.buffer.alloc byte_length = %[[BUF_SZ]] : !vmla.buffer
  // CHECK-NEXT: "vmla.cmp"(%arg0, %arg1, %[[BUF]]) {element_type = i32, predicate = 5 : i32}
  %0 = cmpi "sge", %arg0, %arg1 : tensor<4xi32>
  // CHECK-NEXT: return %[[BUF]]
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: @cmp_f
func @cmp_f(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xi1> attributes { sym_visibility = "private" } {
  // CHECK: %[[BUF_SZ:.+]] = constant 4
  // CHECK-NEXT: %[[BUF:.+]] = vmla.buffer.alloc byte_length = %[[BUF_SZ]] : !vmla.buffer
  // CHECK-NEXT: "vmla.cmp"(%arg0, %arg1, %[[BUF]]) {element_type = f32, predicate = 5 : i32}
  %0 = cmpf "oge", %arg0, %arg1 : tensor<4xf32>
  // CHECK-NEXT: return %[[BUF]]
  return %0 : tensor<4xi1>
}
