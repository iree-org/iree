// RUN: iree-opt --iree-global-opt-cleanup-numeric-narrowing %s | FileCheck %s

// CHECK-LABEL: @remove_inferences
func.func @remove_inferences(%arg0 : tensor<5x3xf32>) -> tensor<5x3xf32> {
  %0 = util.numeric.optional_narrow %arg0 : tensor<5x3xf32> as ui3 {max_value = 5 : ui3, min_value = 5 : ui3}
  // CHECK: return %arg0
  return %0 : tensor<5x3xf32>
}
