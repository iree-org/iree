// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-tensor-to-flow))" | FileCheck %s --check-prefix=EARLY --implicit-check-not=tensor.cast
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-tensor-to-flow,iree-flow-initialize-empty-tensors),canonicalize,iree-stream-conversion)" | FileCheck %s

// An empty conditional result must survive canonicalization after conversion to
// Flow without reintroducing tensor.cast.
// CHECK-LABEL: util.func public @empty_branch
// CHECK: scf.if
// CHECK: %[[EMPTY:.*]] = stream.tensor.empty : tensor<?x8x?x36xf32>
// CHECK: scf.yield %[[EMPTY]],
// EARLY-LABEL: util.func public @empty_branch
// EARLY: flow.tensor.empty : tensor<?x8x?x36xf32>
util.func public @empty_branch(%condition: i1, %other: tensor<?x8x?x36xf32>) -> tensor<?x8x?x36xf32> {
  %result = scf.if %condition -> tensor<?x8x?x36xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty(%c0, %c1) : tensor<?x8x?x36xf32>
    scf.yield %empty : tensor<?x8x?x36xf32>
  } else {
    scf.yield %other : tensor<?x8x?x36xf32>
  }
  util.return %result : tensor<?x8x?x36xf32>
}

// -----

// CHECK-LABEL: util.func public @constant_dim
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[EMPTY:.*]] = stream.tensor.empty : tensor<?x4xf32>{%[[C8]]}
// CHECK: util.return %[[EMPTY]],
// EARLY-LABEL: util.func public @constant_dim
// EARLY: %[[EMPTY:.*]] = tensor.empty() : tensor<8x4xf32>
// EARLY: flow.tensor.reshape %[[EMPTY]] : tensor<8x4xf32> -> tensor<?x4xf32>
util.func public @constant_dim() -> tensor<?x4xf32> {
  %c8 = arith.constant 8 : index
  %empty = tensor.empty(%c8) : tensor<?x4xf32>
  util.return %empty : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: util.func public @mixed_dims
// CHECK-SAME: %[[DIM:.*]]: index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[EMPTY:.*]] = stream.tensor.empty : tensor<?x8x?xf32>{%[[DIM]], %[[C4]]}
// CHECK: util.return %[[EMPTY]],
// EARLY-LABEL: util.func public @mixed_dims
// EARLY: %[[EMPTY:.*]] = tensor.empty(%{{.*}}) : tensor<?x8x4xf32>
// EARLY: flow.tensor.reshape %[[EMPTY]] : tensor<?x8x4xf32>
util.func public @mixed_dims(%dim: index) -> tensor<?x8x?xf32> {
  %c4 = arith.constant 4 : index
  %empty = tensor.empty(%dim, %c4) : tensor<?x8x?xf32>
  util.return %empty : tensor<?x8x?xf32>
}

// -----

// CHECK-LABEL: util.func public @dynamic_dim
// CHECK-SAME: %[[DIM:.*]]: index
// CHECK: %[[EMPTY:.*]] = stream.tensor.empty : tensor<?x4xf32>{%[[DIM]]}
// CHECK: util.return %[[EMPTY]],
// EARLY-LABEL: util.func public @dynamic_dim
// EARLY: tensor.empty(%{{.*}}) : tensor<?x4xf32>
util.func public @dynamic_dim(%dim: index) -> tensor<?x4xf32> {
  %empty = tensor.empty(%dim) : tensor<?x4xf32>
  util.return %empty : tensor<?x4xf32>
}
