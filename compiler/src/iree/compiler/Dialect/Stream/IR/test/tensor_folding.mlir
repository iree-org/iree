// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldTensorImportOp
func.func @FoldTensorImportOp(%arg0: !stream.resource<external>, %arg1: index) -> !stream.resource<external> {
  // CHECK-NOT: stream.tensor.import
  // CHECK-NOT: stream.tensor.export
  // CHECK: return %arg0 : !stream.resource<external>
  %c20 = arith.constant 20 : index
  %0 = stream.tensor.export %arg0 : tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20} -> !hal.buffer_view
  %1 = stream.tensor.import %0 : !hal.buffer_view -> tensor<1x?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  return %1 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @FoldTensorExportOp
func.func @FoldTensorExportOp(%arg0: !hal.buffer_view, %arg1: index) -> !hal.buffer_view {
  // CHECK-NOT: stream.tensor.import
  // CHECK-NOT: stream.tensor.export
  // CHECK: return %arg0 : !hal.buffer_view
  %c20 = arith.constant 20 : index
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  %1 = stream.tensor.export %0 : tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20} -> !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @NofoldTensorExportOpBufferToView
func.func @NofoldTensorExportOpBufferToView(%arg0: !hal.buffer, %arg1: index) -> !hal.buffer_view {
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import
  // CHECK: %[[EXPORT:.+]] = stream.tensor.export %[[IMPORT]]
  // CHECK: return %[[EXPORT]] : !hal.buffer_view
  %c20 = arith.constant 20 : index
  %0 = stream.tensor.import %arg0 : !hal.buffer -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  %1 = stream.tensor.export %0 : tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20} -> !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @KeepTensorExportOpWithDifferingEncodings
func.func @KeepTensorExportOpWithDifferingEncodings(%arg0: !hal.buffer_view, %arg1: index) -> !hal.buffer_view {
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  // CHECK: %[[EXPORT:.+]] = stream.tensor.export %[[IMPORT]] : tensor<1x?x5xf32>{%arg1} in !stream.resource<external>{%c20} -> !hal.buffer_view
  // CHECK: return %[[EXPORT]] : !hal.buffer_view
  %c20 = arith.constant 20 : index
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  %1 = stream.tensor.export %0 : tensor<1x?x5xf32>{%arg1} in !stream.resource<external>{%c20} -> !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @TensorConstantToEmpty
func.func @TensorConstantToEmpty(%arg0: index) -> !stream.resource<constant> {
  // CHECK: %[[EMPTY:.+]] = stream.tensor.empty : tensor<2x0x?xf32>{%arg0} in !stream.resource<constant>
  // CHECK: return %[[EMPTY]]
  // CHECK-NOT: stream.tensor.constant
  %cst = stream.tensor.constant : tensor<2x0x?xf32>{%arg0} in !stream.resource<constant> = dense<> : tensor<2x0x4xf32>
  return %cst : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @TensorConstantToEmptyDynamic
func.func @TensorConstantToEmptyDynamic() -> !stream.resource<constant> {
  // CHECK: %[[EMPTY:.+]] = stream.tensor.empty : tensor<2x?xf32>{%c0} in !stream.resource<constant>
  // CHECK: return %[[EMPTY]]
  // CHECK-NOT: stream.tensor.constant
  %c0 = arith.constant 0 : index
  %cst = stream.tensor.constant : tensor<2x?xf32>{%c0} in !stream.resource<constant> = dense<> : tensor<2x0xf32>
  return %cst : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @TensorConstantToSplat
func.func @TensorConstantToSplat() -> !stream.resource<constant> {
  // CHECK-DAG: %[[CST:.+]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<2x2xf32> : index
  // CHECK: = stream.tensor.splat %[[CST]] : f32 -> tensor<2x2xf32> in !stream.resource<*>{%[[SIZE]]}
  %cst = stream.tensor.constant : tensor<2x2xf32> in !stream.resource<constant> = dense<1.000000e+00> : tensor<2x2xf32>
  return %cst : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @TensorComplexConstantToSplat
func.func @TensorComplexConstantToSplat() -> !stream.resource<constant> {
  // CHECK-DAG: %[[CST:.+]] = complex.constant [2.000000e+00 : f32, 3.000000e+00 : f32] : complex<f32>
  // CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<2x2xcomplex<f32>> : index
  // CHECK: = stream.tensor.splat %[[CST]] : complex<f32> -> tensor<2x2xcomplex<f32>> in !stream.resource<*>{%[[SIZE]]}
  %cst = stream.tensor.constant : tensor<2x2xcomplex<f32>> in !stream.resource<constant> = dense<(2.000000e+00,3.000000e+00)> : tensor<2x2xcomplex<f32>>
  return %cst : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @NarrowSplatPatternI32ToI8
func.func @NarrowSplatPatternI32ToI8() -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %pattern = arith.constant 0xAAAAAAAA : i32
  // CHECK: stream.tensor.splat %c-86_i8 : i8
  %0 = stream.tensor.splat %pattern : i32 -> tensor<2x2xf32> in !stream.resource<*>{%c100}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @NarrowSplatPatternI32ToI16
func.func @NarrowSplatPatternI32ToI16() -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %pattern = arith.constant 0xAABBAABB : i32
  // CHECK: stream.tensor.splat %c-21829_i16 : i16
  %0 = stream.tensor.splat %pattern : i32 -> tensor<2x2xf32> in !stream.resource<*>{%c100}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @NarrowSplatPatternI64ToI8
func.func @NarrowSplatPatternI64ToI8() -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %pattern = arith.constant 0 : i64
  // CHECK: stream.tensor.splat %c0_i8 : i8
  %0 = stream.tensor.splat %pattern : i64 -> tensor<2x2xf32> in !stream.resource<*>{%c100}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @NarrowSplatPatternI64ToI16
func.func @NarrowSplatPatternI64ToI16() -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %pattern = arith.constant 0xAABBAABBAABBAABB : i64
  // CHECK: stream.tensor.splat %c-21829_i16 : i16
  %0 = stream.tensor.splat %pattern : i64 -> tensor<2x2xf32> in !stream.resource<*>{%c100}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @NarrowSplatPatternI64ToI32
func.func @NarrowSplatPatternI64ToI32() -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %pattern = arith.constant 0xAABBCCDDAABBCCDD : i64
  // CHECK: stream.tensor.splat %c12307677_i32
  %0 = stream.tensor.splat %pattern : i64 -> tensor<2x2xf32> in !stream.resource<*>{%c100}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @NarrowSplatPatternBF16
func.func @NarrowSplatPatternBF16() -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %pattern = arith.constant 0.0 : bf16
  // CHECK: stream.tensor.splat %c0_i8 : i8
  %0 = stream.tensor.splat %pattern : bf16 -> tensor<2x2xf32> in !stream.resource<*>{%c100}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @NarrowSplatPatternF32
func.func @NarrowSplatPatternF32() -> !stream.resource<*> {
  %c100 = arith.constant 100 : index
  %pattern = arith.constant 0.0 : f32
  // CHECK: stream.tensor.splat %c0_i8 : i8
  %0 = stream.tensor.splat %pattern : f32 -> tensor<2x2xf32> in !stream.resource<*>{%c100}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @FoldTensorCloneOp
func.func @FoldTensorCloneOp(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  // CHECK-NOT: stream.tensor.clone
  %0 = stream.tensor.clone %arg0 : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<2x2xf32> in !stream.resource<*>{%arg1}
  // CHECK: return %arg0
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @NofoldTensorCloneOp
func.func @NofoldTensorCloneOp(%arg0: !stream.resource<external>, %arg1: index) -> !stream.resource<*> {
  // CHECK: %[[CLONE:.+]] = stream.tensor.clone
  %0 = stream.tensor.clone %arg0 : tensor<2x2xf32> in !stream.resource<external>{%arg1} -> tensor<2x2xf32> in !stream.resource<*>{%arg1}
  // CHECK: return %[[CLONE]] : !stream.resource<*>
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @ElideUnneededTensorClones
func.func @ElideUnneededTensorClones(%arg0: !stream.resource<*>, %arg1: index) -> f32 {
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
