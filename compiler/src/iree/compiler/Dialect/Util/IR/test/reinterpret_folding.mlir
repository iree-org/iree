// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @ReinterpretReinterpretOptimization
func.func @ReinterpretReinterpretOptimization(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: return %arg0
  %0 = util.reinterpret %arg0 : tensor<2xi32> to tensor<2xui32>
  %1 = util.reinterpret %0 : tensor<2xui32> to tensor<2xi32>
  return %1 : tensor<2xi32>
}

// -----

// CHECK-LABEL: @StreamExportReinterpretOptimization
func.func @StreamExportReinterpretOptimization(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<external> {
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.transfer %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<external>{%arg1}
  %0 = stream.async.transfer %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<external>{%arg1}
  %1 = stream.tensor.export %0 : tensor<2xui32> in !stream.resource<external>{%arg1} -> tensor<2xui32>
  %2 = util.reinterpret %1 : tensor<2xui32> to !stream.resource<external>
  // CHECK-NEXT: return %[[RESULT]]
  return %2: !stream.resource<external>
}
