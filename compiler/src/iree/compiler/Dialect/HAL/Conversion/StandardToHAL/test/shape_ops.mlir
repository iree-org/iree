// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: @tensorDim
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view)
util.func public @tensorDim(%arg0: tensor<4x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  // CHECK: %[[DIM:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[1] : index
  %dim = tensor.dim %arg0, %c1 : tensor<4x?xf32>
  util.return %dim : index
}

// -----

// NOTE: we only support ranked tensors, and hal.buffer_view.rank will fold
// immediately to the static value. We check here that the folding happened.

// CHECK: @tensorRank
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view)
util.func public @tensorRank(%arg0: tensor<4x?xf32>) -> index {
  // CHECK-NOT: hal.buffer_view.rank
  // CHECK: %[[RANK:.+]] = arith.constant 2
  %rank = tensor.rank %arg0 : tensor<4x?xf32>
  // CHECK: util.return %[[RANK]]
  util.return %rank : index
}
