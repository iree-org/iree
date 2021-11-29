// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @tensorCastStatic
func @tensorCastStatic(%arg0: !hal.buffer_view) -> tensor<5xi32> {
  // CHECK: hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// CHECK-LABEL: @tensorCastDynamicInput
func @tensorCastDynamicInput(%arg0: tensor<?x3xi32>, %arg1 : index) -> !hal.buffer_view {
  // CHECK: hal.tensor.cast %arg0 : tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  %0 = hal.tensor.cast %arg0 : tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// CHECK-LABEL: @tensorCastDynamicOutput
func @tensorCastDynamicOutput(%arg0: !hal.buffer_view, %arg1 : index) -> tensor<?x3xi32> {
  // CHECK: hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<?x3xi32>{%arg1}
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<?x3xi32>{%arg1}
  return %0 : tensor<?x3xi32>
}
