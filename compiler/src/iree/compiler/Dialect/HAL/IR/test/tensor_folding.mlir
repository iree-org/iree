// RUN: iree-opt --split-input-file --canonicalize -cse %s | FileCheck %s

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

// CHECK-LABEL: @FoldConsecutiveTransientsSameStorage
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[STORAGE:.+]]: !hal.buffer)
util.func public @FoldConsecutiveTransientsSameStorage(%arg0: tensor<4xf32>, %storage: !hal.buffer) -> tensor<4xf32> {
  // CHECK-NOT: hal.tensor.transients
  %0 = hal.tensor.transients %arg0 : tensor<4xf32> from %storage : !hal.buffer
  // CHECK: %[[RESULT:.+]] = hal.tensor.transients %[[ARG0]] : tensor<4xf32> from %[[STORAGE]] : !hal.buffer
  %1 = hal.tensor.transients %0 : tensor<4xf32> from %storage : !hal.buffer
  // CHECK: util.return %[[RESULT]] : tensor<4xf32>
  util.return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @FoldConsecutiveTransientsDifferentStorage
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[STORAGE1:.+]]: !hal.buffer, %[[STORAGE2:.+]]: !hal.buffer)
util.func public @FoldConsecutiveTransientsDifferentStorage(%arg0: tensor<4xf32>, %storage1: !hal.buffer, %storage2: !hal.buffer) -> tensor<4xf32> {
  // CHECK-NOT: hal.tensor.transients %[[ARG0]] : tensor<4xf32> from %[[STORAGE1]]
  %0 = hal.tensor.transients %arg0 : tensor<4xf32> from %storage1 : !hal.buffer
  // CHECK: %[[RESULT:.+]] = hal.tensor.transients %[[ARG0]] : tensor<4xf32> from %[[STORAGE2]] : !hal.buffer
  %1 = hal.tensor.transients %0 : tensor<4xf32> from %storage2 : !hal.buffer
  // CHECK: util.return %[[RESULT]] : tensor<4xf32>
  util.return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @FoldConsecutiveTransientsDynamic
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x4xf32>, %[[DIM:.+]]: index, %[[STORAGE1:.+]]: !hal.buffer, %[[STORAGE2:.+]]: !hal.buffer)
util.func public @FoldConsecutiveTransientsDynamic(%arg0: tensor<?x4xf32>, %dim: index, %storage1: !hal.buffer, %storage2: !hal.buffer) -> tensor<?x4xf32> {
  // CHECK-NOT: hal.tensor.transients %[[ARG0]] : tensor<?x4xf32>{%[[DIM]]} from %[[STORAGE1]]
  %0 = hal.tensor.transients %arg0 : tensor<?x4xf32>{%dim} from %storage1 : !hal.buffer
  // CHECK: %[[RESULT:.+]] = hal.tensor.transients %[[ARG0]] : tensor<?x4xf32>{%[[DIM]]} from %[[STORAGE2]] : !hal.buffer
  %1 = hal.tensor.transients %0 : tensor<?x4xf32>{%dim} from %storage2 : !hal.buffer
  // CHECK: util.return %[[RESULT]] : tensor<?x4xf32>
  util.return %1 : tensor<?x4xf32>
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
