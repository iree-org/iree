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

// Import wait fences must not be folded away.

// CHECK-LABEL: @noFoldTensorImportExportWithWait
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view, %[[WAIT:.+]]: !hal.fence)
util.func public @noFoldTensorImportExportWithWait(%arg0: !hal.buffer_view, %wait: !hal.fence) -> !hal.buffer_view {
  // CHECK: %[[IMPORT:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.import wait(%wait) => %arg0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %[[IMPORT]] : tensor<5xi32> -> !hal.buffer_view
  %1 = hal.tensor.export %0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK: util.return %[[EXPORT]] : !hal.buffer_view
  util.return %1 : !hal.buffer_view
}

// -----

// Import consumes must not be folded away.

// CHECK-LABEL: @noFoldTensorImportExportWithConsume
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view)
util.func public @noFoldTensorImportExportWithConsume(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK: %[[IMPORT:.+]] = hal.tensor.import consume %[[ARG0]] : !hal.buffer_view -> tensor<5xi32>
  %0 = hal.tensor.import consume %arg0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %[[IMPORT]] : tensor<5xi32> -> !hal.buffer_view
  %1 = hal.tensor.export %0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK: util.return %[[EXPORT]] : !hal.buffer_view
  util.return %1 : !hal.buffer_view
}

// -----

// Import wait fences must not be folded away from export/import pairs.

// CHECK-LABEL: @noFoldTensorExportImportWithWait
// CHECK-SAME: (%[[ARG0:.+]]: tensor<5xi32>, %[[WAIT:.+]]: !hal.fence)
util.func public @noFoldTensorExportImportWithWait(%arg0: tensor<5xi32>, %wait: !hal.fence) -> tensor<5xi32> {
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %[[ARG0]] : tensor<5xi32> -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK: %[[IMPORT:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[EXPORT]] : !hal.buffer_view -> tensor<5xi32>
  %1 = hal.tensor.import wait(%wait) => %0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: util.return %[[IMPORT]] : tensor<5xi32>
  util.return %1 : tensor<5xi32>
}

// -----

// Import consumes must not be folded away from export/import pairs.

// CHECK-LABEL: @noFoldTensorExportImportWithConsume
// CHECK-SAME: (%[[ARG0:.+]]: tensor<5xi32>)
util.func public @noFoldTensorExportImportWithConsume(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %[[ARG0]] : tensor<5xi32> -> !hal.buffer_view
  %0 = hal.tensor.export %arg0 : tensor<5xi32> -> !hal.buffer_view
  // CHECK: %[[IMPORT:.+]] = hal.tensor.import consume %[[EXPORT]] : !hal.buffer_view -> tensor<5xi32>
  %1 = hal.tensor.import consume %0 : !hal.buffer_view -> tensor<5xi32>
  // CHECK: util.return %[[IMPORT]] : tensor<5xi32>
  util.return %1 : tensor<5xi32>
}

// -----

// Import with byte_offset must not fold — the import is a subview.

// CHECK-LABEL: @noFoldTensorImportExportWithOffset
util.func public @noFoldTensorImportExportWithOffset(%arg0: !hal.buffer_view, %arg1: index) -> !hal.buffer_view {
  // CHECK: hal.tensor.import
  %0 = hal.tensor.import %arg0 offset(%arg1) : !hal.buffer_view -> tensor<5xi32>
  // CHECK: hal.tensor.export
  %1 = hal.tensor.export %0 : tensor<5xi32> -> !hal.buffer_view
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

// CHECK-LABEL: @propagateImportDimsFromConsumers
// CHECK-SAME: (%[[BV:.+]]: !hal.buffer_view, %[[BUF:.+]]: !hal.buffer)
util.func public @propagateImportDimsFromConsumers(%bv: !hal.buffer_view, %buf: !hal.buffer) -> tensor<?x?xf32> {
  %d0 = hal.buffer_view.dim<%bv : !hal.buffer_view>[0] : index
  %d1 = hal.buffer_view.dim<%bv : !hal.buffer_view>[1] : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: hal.tensor.import %[[BV]] : !hal.buffer_view -> tensor<?x?xf32>{%[[C4]], %[[C8]]}
  %t = hal.tensor.import %bv : !hal.buffer_view -> tensor<?x?xf32>{%d0, %d1}
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %aliased = hal.tensor.alias %t : tensor<?x?xf32>{%c4, %c8} to %buf : !hal.buffer
  util.return %aliased : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @propagateImportDimsPartial
// CHECK-SAME: (%[[BV:.+]]: !hal.buffer_view, %[[BUF:.+]]: !hal.buffer)
util.func public @propagateImportDimsPartial(%bv: !hal.buffer_view, %buf: !hal.buffer) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[D0:.+]] = hal.buffer_view.dim<%[[BV]] : !hal.buffer_view>[0] : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  %d0 = hal.buffer_view.dim<%bv : !hal.buffer_view>[0] : index
  %d1 = hal.buffer_view.dim<%bv : !hal.buffer_view>[1] : index
  // CHECK: hal.tensor.import %[[BV]] : !hal.buffer_view -> tensor<?x?xf32>{%[[D0]], %[[C8]]}
  %t = hal.tensor.import %bv : !hal.buffer_view -> tensor<?x?xf32>{%d0, %d1}
  %c8 = arith.constant 8 : index
  %aliased = hal.tensor.alias %t : tensor<?x?xf32>{%d0, %c8} to %buf : !hal.buffer
  util.return %aliased : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @propagateImportDimsConflictingConsumers
// CHECK-SAME: (%[[BV:.+]]: !hal.buffer_view, %[[BUF0:.+]]: !hal.buffer, %[[BUF1:.+]]: !hal.buffer)
util.func public @propagateImportDimsConflictingConsumers(%bv: !hal.buffer_view, %buf0: !hal.buffer, %buf1: !hal.buffer) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK-DAG: %[[D0:.+]] = hal.buffer_view.dim<%[[BV]] : !hal.buffer_view>[0] : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  %d0 = hal.buffer_view.dim<%bv : !hal.buffer_view>[0] : index
  %d1 = hal.buffer_view.dim<%bv : !hal.buffer_view>[1] : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  // CHECK: hal.tensor.import %[[BV]] : !hal.buffer_view -> tensor<?x?xf32>{%[[D0]], %[[C8]]}
  %t = hal.tensor.import %bv : !hal.buffer_view -> tensor<?x?xf32>{%d0, %d1}
  %aliased0 = hal.tensor.alias %t : tensor<?x?xf32>{%c4, %c8} to %buf0 : !hal.buffer
  %aliased1 = hal.tensor.alias %t : tensor<?x?xf32>{%c5, %c8} to %buf1 : !hal.buffer
  util.return %aliased0, %aliased1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @doNotPropagateImportDimsFromNestedConsumer
// CHECK-SAME: (%[[BV:.+]]: !hal.buffer_view, %[[BUF:.+]]: !hal.buffer, %[[COND:.+]]: i1)
util.func public @doNotPropagateImportDimsFromNestedConsumer(%bv: !hal.buffer_view, %buf: !hal.buffer, %cond: i1) -> tensor<?xf32> {
  // CHECK-DAG: %[[D0:.+]] = hal.buffer_view.dim<%[[BV]] : !hal.buffer_view>[0] : index
  %d0 = hal.buffer_view.dim<%bv : !hal.buffer_view>[0] : index
  // CHECK: hal.tensor.import %[[BV]] : !hal.buffer_view -> tensor<?xf32>{%[[D0]]}
  %t = hal.tensor.import %bv : !hal.buffer_view -> tensor<?xf32>{%d0}
  %r = scf.if %cond -> tensor<?xf32> {
    %c4 = arith.constant 4 : index
    %aliased = hal.tensor.alias %t : tensor<?xf32>{%c4} to %buf : !hal.buffer
    scf.yield %aliased : tensor<?xf32>
  } else {
    scf.yield %t : tensor<?xf32>
  }
  util.return %r : tensor<?xf32>
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
