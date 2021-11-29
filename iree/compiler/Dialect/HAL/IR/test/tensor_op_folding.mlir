// RUN: iree-opt -split-input-file -canonicalize -cse %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @tensorCastMatchingTypeFolds
func @tensorCastMatchingTypeFolds(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK-NOT: hal.tensor.cast
  // CHECK: return %arg0 : !hal.buffer_view
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorCastPassthroughFolds
func @tensorCastPassthroughFolds(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK-NOT: hal.tensor.cast
  // CHECK: return %arg0 : !hal.buffer_view
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5xi32>
  %1 = hal.tensor.cast %0 : tensor<5xi32> -> !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorCastThroughDifferentTypesFolds
func @tensorCastThroughDifferentTypesFolds(%arg0: !hal.buffer_view) -> !hal.buffer {
  // CHECK: %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> !hal.buffer
  // CHECK: return %0 : !hal.buffer
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5xi32>
  %1 = hal.tensor.cast %0 : tensor<5xi32> -> !hal.buffer
  return %1 : !hal.buffer
}

// -----

// CHECK-LABEL: @tensorCastFoldingPreservesDims
func @tensorCastFoldingPreservesDims(%arg0: !hal.buffer_view, %arg1 : index) -> tensor<?x3xi32> {
  // CHECK: hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<?x3xi32>{%arg1}
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> !hal.buffer
  %1 = hal.tensor.cast %0 : !hal.buffer -> tensor<?x3xi32>{%arg1}
  return %1 : tensor<?x3xi32>
}
