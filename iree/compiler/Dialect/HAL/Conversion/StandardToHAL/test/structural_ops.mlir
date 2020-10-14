// RUN: iree-opt -split-input-file -iree-convert-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: func @tensorIO(%arg0: !hal.buffer) -> !hal.buffer
func @tensorIO(%arg0 : tensor<1x1xi32>) -> tensor<1x1xi32> {
  // CHECK-NEXT: br ^bb1(%arg0 : !hal.buffer)
  br ^bb1(%arg0 : tensor<1x1xi32>)
// CHECK-NEXT: ^bb1(%[[BB0:.+]]: !hal.buffer)
^bb1(%0 : tensor<1x1xi32>):
  // CHECK-NEXT: return %[[BB0]] : !hal.buffer
  return %0 : tensor<1x1xi32>
}
