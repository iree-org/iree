// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-lower-bounds))" \
// RUN:   %s | FileCheck %s

// A static constant can carry an all-unknown bounds encoding. Strip it from
// both the result and the value attribute without changing the payload.
// CHECK-LABEL: @constant_bounds
// CHECK-SAME: () -> tensor<4xf32>
// CHECK-NEXT: %[[C:.+]] = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
// CHECK-NEXT: return %[[C]] : tensor<4xf32>
func.func @constant_bounds() -> tensor<4xf32, #stablehlo.bounds<?>> {
  %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32, #stablehlo.bounds<?>>
  return %0 : tensor<4xf32, #stablehlo.bounds<?>>
}

// -----

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

// A `?` bound generates no assumption.
// CHECK-LABEL: @partial_bounds
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x8x?xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-DAG: %[[A0:.+]] = util.assume.int %[[D0]]<umax = 16> : index
// CHECK: %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
// CHECK-NOT: util.assume.int %[[D2]]
// CHECK: flow.tensor.tie_shape %[[ARG0]] : tensor<?x8x?xf32>{%[[A0]], %[[D2]]}
func.func @partial_bounds(%arg0: tensor<?x8x?xf32, #stablehlo.bounds<16, ?, ?>>) -> tensor<?x8x?xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<?x8x?xf32, #stablehlo.bounds<16, ?, ?>>) -> tensor<?x8x?xf32>
  return %0 : tensor<?x8x?xf32>
}

// -----

// Bounds on multiple dimensions must survive on both arguments and results.
// CHECK-LABEL: @rank3_two_bounds
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x5x?xf32>) -> tensor<?x5x?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-DAG: %[[A0:.+]] = util.assume.int %[[D0]]<umax = 16> : index
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

// -----

// Every original use must receive the tied value, including repeated operands.
// CHECK-LABEL: @multiple_users
// CHECK-SAME: %[[ARG:.+]]: tensor<?xf32>
// CHECK: %[[ARG_TIED:.+]] = flow.tensor.tie_shape %[[ARG]]
// CHECK: %[[ABS:.+]] = stablehlo.abs %[[ARG_TIED]]
// CHECK: %[[DIM:.+]] = tensor.dim %[[ABS]],
// CHECK: %[[BOUND:.+]] = util.assume.int %[[DIM]]<umax = 8>
// CHECK: %[[TIED:.+]] = flow.tensor.tie_shape %[[ABS]] : tensor<?xf32>{%[[BOUND]]}
// CHECK: %[[NEG:.+]] = stablehlo.negate %[[TIED]]
// CHECK: %[[SUM:.+]] = stablehlo.add %[[TIED]], %[[TIED]]
// CHECK: return %[[NEG]], %[[SUM]]
func.func @multiple_users(%arg0: tensor<?xf32, #stablehlo.bounds<8>>)
    -> (tensor<?xf32>, tensor<?xf32>) {
  %0 = stablehlo.abs %arg0 : tensor<?xf32, #stablehlo.bounds<8>>
  %1 = stablehlo.negate %0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<?xf32>
  %2 = stablehlo.add %0, %0 : (tensor<?xf32, #stablehlo.bounds<8>>, tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<?xf32>
  return %1, %2 : tensor<?xf32>, tensor<?xf32>
}

// -----

// Assumptions for captured values must dominate uses in either region.
// CHECK-LABEL: @nested_uses
// CHECK-SAME: %[[ARG:.+]]: tensor<?xf32>
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG]],
// CHECK: %[[BOUND:.+]] = util.assume.int %[[DIM]]<umax = 8>
// CHECK: %[[TIED:.+]] = flow.tensor.tie_shape %[[ARG]] : tensor<?xf32>{%[[BOUND]]}
// CHECK: scf.if
// CHECK: stablehlo.abs %[[TIED]]
// CHECK: } else {
// CHECK: stablehlo.negate %[[TIED]]
func.func @nested_uses(%arg0: tensor<?xf32, #stablehlo.bounds<8>>, %cond: i1)
    -> tensor<?xf32> {
  %result = scf.if %cond -> tensor<?xf32> {
    %0 = stablehlo.abs %arg0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<?xf32>
    scf.yield %0 : tensor<?xf32>
  } else {
    %0 = stablehlo.negate %arg0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<?xf32>
    scf.yield %0 : tensor<?xf32>
  }
  return %result : tensor<?xf32>
}
