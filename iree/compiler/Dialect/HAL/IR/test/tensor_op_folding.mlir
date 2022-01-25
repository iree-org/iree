// RUN: iree-opt -split-input-file -canonicalize -cse %s | iree-opt -split-input-file | FileCheck %s

// CHECK-LABEL: @foldTensorImportExport
func @foldTensorImportExport(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK-NOT: hal.tensor.import
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK-NOT: hal.tensor.export
  %1 = hal.tensor.export %0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK: return %arg0 : !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

// TODO(benvanik): add a canonicalizer to take buffer_view -> buffer and turn
// it into a hal.buffer_view.buffer op and buffer -> buffer_view into a
// hal.buffer_view.create.
// For now we just don't fold.

// CHECK-LABEL: @foldTensorImportExportTypeMismatch
func @foldTensorImportExportTypeMismatch(%arg0: !hal.buffer_view) -> !hal.buffer {
  // CHECK: hal.tensor.import
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: hal.tensor.export
  %1 = hal.tensor.export %0 : tensor<5xi32> -> !hal.buffer
  return %1 : !hal.buffer
}

// -----

// CHECK-LABEL: @foldTensorExportImport
func @foldTensorExportImport(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  // CHECK-NOT: hal.tensor.export
  %0 = hal.tensor.export %arg0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK-NOT: hal.tensor.import
  %1 = hal.tensor.import %0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: return %arg0 : tensor<5xi32>
  return %1 : tensor<5xi32>
}
