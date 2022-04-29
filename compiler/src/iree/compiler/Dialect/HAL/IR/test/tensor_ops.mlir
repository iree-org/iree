// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | FileCheck %s

// CHECK-LABEL: @tensorImportStatic
func.func @tensorImportStatic(%arg0: !hal.buffer_view) -> tensor<5xi32> {
  // CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// -----

// CHECK-LABEL: @tensorImportDynamic
func.func @tensorImportDynamic(%arg0: !hal.buffer_view, %arg1 : index) -> tensor<?x3xi32> {
  // CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x3xf32> as tensor<?x3xi32>{%arg1}
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x3xf32> as tensor<?x3xi32>{%arg1}
  return %0 : tensor<?x3xi32>
}

// -----

// CHECK-LABEL: @tensorExportDynamic
func.func @tensorExportDynamic(%arg0: tensor<?x3xi32>, %arg1 : index) -> !hal.buffer_view {
  // CHECK: hal.tensor.export %arg0 : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorExportInPlace
func.func @tensorExportInPlace(%arg0: tensor<?x3xi32>, %arg1 : index, %arg2: !hal.buffer) -> !hal.buffer_view {
  // CHECK: hal.tensor.export %arg0 into %arg2 : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 into %arg2 : tensor<?x3xf32> as tensor<?x3xi32>{%arg1} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}
