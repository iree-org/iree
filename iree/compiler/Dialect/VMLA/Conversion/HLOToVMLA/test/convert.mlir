// RUN: iree-opt -split-input-file -iree-vmla-conversion %s | IreeFileCheck %s

// CHECK-LABEL: func @basic
func @basic(%arg0 : tensor<5xf32>) -> (tensor<5xi32>) attributes { sym_visibility = "private" } {
  // CHECK: vmla.convert
  %0 = "mhlo.convert"(%arg0) : (tensor<5xf32>) -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// CHECK-LABEL: func @noop
func @noop(%arg0 : tensor<?xf32>) -> (tensor<5xf32>) attributes { sym_visibility = "private" } {
  // CHECK: return %arg0
  %0 = "mhlo.convert"(%arg0) : (tensor<?xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
