// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @TensorConstantToSplat
func @TensorConstantToSplat() -> !stream.resource<constant> {
  // CHECK-DAG: %[[CST:.+]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<2x2xf32> : index
  // CHECK: = stream.tensor.splat %[[CST]] : f32 -> tensor<2x2xf32> in !stream.resource<*>{%[[SIZE]]}
  %cst = stream.tensor.constant : tensor<2x2xf32> in !stream.resource<constant> = dense<1.000000e+00> : tensor<2x2xf32>
  return %cst : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldTensorCloneOp
func @FoldTensorCloneOp(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  // CHECK-NOT: stream.tensor.clone
  %0 = stream.tensor.clone %arg0 : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<2x2xf32> in !stream.resource<*>{%arg1}
  // CHECK: return %arg0
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ElideUnneededTensorClones
func @ElideUnneededTensorClones(%arg0: !stream.resource<*>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.tensor.clone
  %0 = stream.tensor.clone %arg0 : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<2x2xf32> in !stream.resource<*>{%arg1}
  // CHECK: %[[T0:.+]] = stream.async.transfer %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<staging>{%arg1}
  %1 = stream.async.transfer %0 : !stream.resource<*>{%arg1} -> !stream.resource<staging>{%arg1}
  // CHECK: %[[T1:.+]] = stream.tensor.load %[[T0]][%c0, %c0] : tensor<2x2xf32> in !stream.resource<staging>{%arg1} -> f32
  %2 = stream.tensor.load %1[%c0, %c0] : tensor<2x2xf32> in !stream.resource<staging>{%arg1} -> f32
  // CHECK: return %[[T1]]
  return %2 : f32
}
