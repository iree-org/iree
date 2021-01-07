// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @reshape_bypass
func private @reshape_bypass(%arg0 : tensor<3x2xi32>) -> tensor<6xi32> {
  // CHECK-NEXT: return %arg0
  %0 = "mhlo.reshape"(%arg0) : (tensor<3x2xi32>) -> tensor<6xi32>
  return %0 : tensor<6xi32>
}

// -----

// CHECK-LABEL: @reshape_copy
func private @reshape_copy(%arg0 : tensor<3x2xi32>) -> (tensor<3x2xi32>, tensor<6xi32>) {
  // CHECK-NEXT: %0 = vmla.buffer.clone %arg0 : !vmla.buffer
  %0 = "mhlo.reshape"(%arg0) : (tensor<3x2xi32>) -> tensor<6xi32>
  // CHECK-NEXT: return %arg0, %0
  return %arg0, %0 : tensor<3x2xi32>, tensor<6xi32>
}
