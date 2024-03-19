// RUN: iree-opt --split-input-file --canonicalize -cse %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @foldTensorImportExport
util.func public @foldTensorImportExport(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK-NOT: hal.tensor.import
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK-NOT: hal.tensor.export
  %1 = hal.tensor.export %0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK: util.return %arg0 : !hal.buffer_view
  util.return %1 : !hal.buffer_view
}

// -----

// TODO(benvanik): add a canonicalizer to take buffer_view -> buffer and turn
// it into a hal.buffer_view.buffer op and buffer -> buffer_view into a
// hal.buffer_view.create.
// For now we just don't fold.

// CHECK-LABEL: @foldTensorImportExportTypeMismatch
util.func public @foldTensorImportExportTypeMismatch(%arg0: !hal.buffer_view) -> !hal.buffer {
  // CHECK: hal.tensor.import
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: hal.tensor.export
  %1 = hal.tensor.export %0 : tensor<5xi32> -> !hal.buffer
  util.return %1 : !hal.buffer
}

// -----

// CHECK-LABEL: @foldTensorExportImport
util.func public @foldTensorExportImport(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  // CHECK-NOT: hal.tensor.export
  %0 = hal.tensor.export %arg0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK-NOT: hal.tensor.import
  %1 = hal.tensor.import %0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: util.return %arg0 : tensor<5xi32>
  util.return %1 : tensor<5xi32>
}

// -----

// CHECK-LABEL: @DeduplicateTensorBarrierSources
// CHECK-SAME: (%[[ARG0:.+]]: tensor<5xi32>, %[[ARG1:.+]]: tensor<6xi32>, %[[FENCE:.+]]: !hal.fence)
util.func public @DeduplicateTensorBarrierSources(%arg0: tensor<5xi32>, %arg1: tensor<6xi32>, %fence: !hal.fence) -> (tensor<5xi32>, tensor<6xi32>, tensor<5xi32>) {
  // CHECK: %[[RESULTS:.+]]:2 = hal.tensor.barrier join(%[[ARG0]], %[[ARG1]] : tensor<5xi32>, tensor<6xi32>) => %[[FENCE]] : !hal.fence
  %0:3 = hal.tensor.barrier join(%arg0, %arg1, %arg0 : tensor<5xi32>, tensor<6xi32>, tensor<5xi32>) => %fence : !hal.fence
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[RESULTS]]#0 : tensor<5xi32>, tensor<6xi32>, tensor<5xi32>
  util.return %0#0, %0#1, %0#2 : tensor<5xi32>, tensor<6xi32>, tensor<5xi32>
}
