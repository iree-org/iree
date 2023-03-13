// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @tensorImportStatic
func.func @tensorImportStatic(%arg0: !hal.buffer_view) -> tensor<5xi32> {
  // CHECK: hal.tensor.import %arg0 "hello" : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.import %arg0 "hello" : !hal.buffer_view -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// -----

// CHECK-LABEL: @tensorImportDynamic
func.func @tensorImportDynamic(%arg0: !hal.buffer_view, %arg1: index) -> tensor<?x3xi32> {
  // CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x3xf32> as tensor<?x3xi32>{%arg1}
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x3xf32> as tensor<?x3xi32>{%arg1}
  return %0 : tensor<?x3xi32>
}

// -----

// CHECK-LABEL: @tensorImportAsync
func.func @tensorImportAsync(%arg0: !hal.buffer_view, %arg1: !hal.fence) -> tensor<5xi32> {
  // CHECK: hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// -----

// CHECK-LABEL: @tensorExportDynamic
func.func @tensorExportDynamic(%arg0: tensor<?x3xi32>, %arg1: index) -> !hal.buffer_view {
  // CHECK: hal.tensor.export %arg0 "goodbye" : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 "goodbye" : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorExportInPlace
func.func @tensorExportInPlace(%arg0: tensor<?x3xi32>, %arg1: index, %arg2: !hal.buffer) -> !hal.buffer_view {
  // CHECK: hal.tensor.export %arg0 into %arg2 : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 into %arg2 : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorBarrier
func.func @tensorBarrier(%arg0: tensor<3xf32>, %arg1: tensor<4xf32>, %arg2: !hal.fence) -> (tensor<3xf32>, tensor<4xf32>) {
  // CHECK: :2 = hal.tensor.barrier join(%arg0, %arg1 : tensor<3xf32>, tensor<4xf32>) => %arg2 : !hal.fence
  %0:2 = hal.tensor.barrier join(%arg0, %arg1 : tensor<3xf32>, tensor<4xf32>) => %arg2 : !hal.fence
  return %0#0, %0#1 : tensor<3xf32>, tensor<4xf32>
}
