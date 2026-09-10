// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-lower-bounds))" \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @bounds_in_signature
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>) -> tensor<?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[ASSUMED:.+]] = util.assume.int %[[DIM]]<umax = 8> : index
// CHECK: %[[TIED:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<?xf32>{%[[ASSUMED]]}
// CHECK: %[[ABS:.+]] = stablehlo.abs %[[TIED]] : tensor<?xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[ABS]], %[[C0]]
// CHECK: %[[ASSUMED1:.+]] = util.assume.int %[[DIM1]]<umax = 8> : index
// CHECK: %[[TIED1:.+]] = flow.tensor.tie_shape %[[ABS]] : tensor<?xf32>{%[[ASSUMED1]]}
// CHECK: return %[[TIED1]] : tensor<?xf32>
func.func @bounds_in_signature(%arg0: tensor<?xf32, #stablehlo.bounds<8>>)
    -> tensor<?xf32, #stablehlo.bounds<8>> {
  %0 = stablehlo.abs %arg0 : tensor<?xf32, #stablehlo.bounds<8>>
  return %0 : tensor<?xf32, #stablehlo.bounds<8>>
}

// -----

// A `?` bound leaves that dimension unassumed; static dims get nothing.
// CHECK-LABEL: @partial_bounds
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x8x?xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[A0:.+]] = util.assume.int %[[D0]]<umax = 16> : index
// CHECK: %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
// CHECK-NOT: util.assume.int %[[D2]]
// CHECK: flow.tensor.tie_shape %[[ARG0]] : tensor<?x8x?xf32>{%[[A0]], %[[D2]]}
func.func @partial_bounds(%arg0: tensor<?x8x?xf32, #stablehlo.bounds<16, ?, ?>>) -> tensor<?x8x?xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<?x8x?xf32, #stablehlo.bounds<16, ?, ?>>) -> tensor<?x8x?xf32>
  return %0 : tensor<?x8x?xf32>
}

// -----

// Two bounded dims on a rank-3 argument, and a bounded result of a second
// op. Each dim gets its own dim, assume and tie.
// CHECK-LABEL: @rank3_two_bounds
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[A0:.+]] = util.assume.int %[[D0]]<umax = 16> : index
// CHECK: %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
// CHECK: %[[A2:.+]] = util.assume.int %[[D2]]<umax = 9> : index
// CHECK: %[[TIED:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<?x5x?xf32>{%[[A0]], %[[A2]]}
// CHECK: %[[ABS:.+]] = stablehlo.abs %[[TIED]]
// CHECK: %[[NEG:.+]] = stablehlo.negate %[[ABS]]
// CHECK: %[[N0:.+]] = tensor.dim %[[NEG]], %[[C0]]
// CHECK: %[[NA0:.+]] = util.assume.int %[[N0]]<umax = 16> : index
// CHECK: %[[N2:.+]] = tensor.dim %[[NEG]], %[[C2]]
// CHECK: %[[NA2:.+]] = util.assume.int %[[N2]]<umax = 9> : index
// CHECK: %[[NTIED:.+]] = flow.tensor.tie_shape %[[NEG]] : tensor<?x5x?xf32>{%[[NA0]], %[[NA2]]}
// CHECK: return %[[NTIED]]
func.func @rank3_two_bounds(%arg0: tensor<?x5x?xf32, #stablehlo.bounds<16, ?, 9>>)
    -> tensor<?x5x?xf32, #stablehlo.bounds<16, ?, 9>> {
  %0 = stablehlo.abs %arg0 : (tensor<?x5x?xf32, #stablehlo.bounds<16, ?, 9>>) -> tensor<?x5x?xf32>
  %1 = stablehlo.negate %0 : (tensor<?x5x?xf32>) -> tensor<?x5x?xf32, #stablehlo.bounds<16, ?, 9>>
  return %1 : tensor<?x5x?xf32, #stablehlo.bounds<16, ?, 9>>
}
