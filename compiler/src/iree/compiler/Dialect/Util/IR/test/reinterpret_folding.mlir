// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @ReinterpretReinterpretOptimization
func.func @ReinterpretReinterpretOptimization(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: return %arg0
  %0 = util.reinterpret %arg0 : tensor<2xi32> to tensor<2xui32>
  %1 = util.reinterpret %0 : tensor<2xui32> to tensor<2xi32>
  return %1 : tensor<2xi32>
}
