// RUN: iree-opt --split-input-file --iree-global-opt-expand-tensor-shapes %s | FileCheck %s

// Tests that global tensor loads also load their dynamic dimensions.

//      CHECK: util.global private mutable @loadedGlobal : tensor<4x?x?x2xf32>
// CHECK-NEXT: util.global private mutable @loadedGlobal__d1 : index
// CHECK-NEXT: util.global private mutable @loadedGlobal__d2 : index
util.global private mutable @loadedGlobal : tensor<4x?x?x2xf32>

// CHECK-LABEL: @globalLoad
func.func @globalLoad() {
  // CHECK-NEXT: %[[TENSOR:.+]] = util.global.load @loadedGlobal : tensor<4x?x?x2xf32>
  // CHECK-NEXT: %[[D1:.+]] = util.global.load @loadedGlobal__d1 : index
  // CHECK-NEXT: %[[D2:.+]] = util.global.load @loadedGlobal__d2 : index
  // CHECK-NEXT: %[[TIED:.+]] = flow.tensor.tie_shape %[[TENSOR]] : tensor<4x?x?x2xf32>{%[[D1]], %[[D2]]}
  %0 = util.global.load @loadedGlobal : tensor<4x?x?x2xf32>
  // CHECK-NEXT: util.optimization_barrier %[[TIED]]
  util.optimization_barrier %0 : tensor<4x?x?x2xf32>
  return
}

// -----

// Tests that global tensor stores also store their dynamic dimensions.

//      CHECK: util.global private mutable @storedGlobal : tensor<4x?x?x2xf32>
// CHECK-NEXT: util.global private mutable @storedGlobal__d1 : index
// CHECK-NEXT: util.global private mutable @storedGlobal__d2 : index
util.global private mutable @storedGlobal : tensor<4x?x?x2xf32>

// CHECK-LABEL: @globalStore
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?x?x2xf32>, %[[D1:.+]]: index, %[[D2:.+]]: index)
func.func @globalStore(%arg0: tensor<4x?x?x2xf32>) {
  // CHECK-NEXT: %[[TIED:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<4x?x?x2xf32>{%[[D1]], %[[D2]]}
  // CHECK-NEXT: util.global.store %[[ARG0]], @storedGlobal : tensor<4x?x?x2xf32>
  // CHECK-NEXT: util.global.store %[[D1]], @storedGlobal__d1 : index
  // CHECK-NEXT: util.global.store %[[D2]], @storedGlobal__d2 : index
  util.global.store %arg0, @storedGlobal : tensor<4x?x?x2xf32>
  return
}

// -----

// Tests that function arguments are expanded into (tensor, dynamic dims...).

// CHECK-LABEL: @funcArgs
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?x?x2xf32>, %[[ARG0_D1:.+]]: index, %[[ARG0_D2:.+]]: index,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<?xi32>, %[[ARG1_D0:.+]]: index)
func.func @funcArgs(%arg0: tensor<4x?x?x2xf32>, %arg1: tensor<?xi32>) {
  // CHECK-NEXT: %[[TIED_ARG0:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<4x?x?x2xf32>{%[[ARG0_D1]], %[[ARG0_D2]]}
  // CHECK-NEXT: %[[TIED_ARG1:.+]] = flow.tensor.tie_shape %[[ARG1]] : tensor<?xi32>{%[[ARG1_D0]]}

  // CHECK-NEXT: util.optimization_barrier %[[TIED_ARG0]]
  util.optimization_barrier %arg0 : tensor<4x?x?x2xf32>
  // CHECK-NEXT: util.optimization_barrier %[[TIED_ARG1]]
  util.optimization_barrier %arg1 : tensor<?xi32>

  return
}

// -----

// Tests that function results are expanded into (tensor, dynamic dims...).

// CHECK-LABEL: @funcResults
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?x?x2xf32>, %[[ARG0_D1:.+]]: index, %[[ARG0_D2:.+]]: index,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<?xi32>, %[[ARG1_D0:.+]]: index)
func.func @funcResults(%arg0: tensor<4x?x?x2xf32>, %arg1: tensor<?xi32>) -> (tensor<4x?x?x2xf32>, tensor<?xi32>) {
  // CHECK-NEXT: %[[TIED_ARG0:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<4x?x?x2xf32>{%[[ARG0_D1]], %[[ARG0_D2]]}
  // CHECK-NEXT: %[[TIED_ARG1:.+]] = flow.tensor.tie_shape %[[ARG1]] : tensor<?xi32>{%[[ARG1_D0]]}

  // NOTE: we return %arg0/%arg1 instead of the tied ones - this helps the ties
  // get dropped early when they aren't needed.
  // CHECK-NEXT: return %[[ARG0]], %[[ARG0_D1]], %[[ARG0_D2]], %[[ARG1]], %[[ARG1_D0]]
  return %arg0, %arg1 : tensor<4x?x?x2xf32>, tensor<?xi32>
}

// -----

// Tests that function calls have their args and results expanded.

// CHECK-LABEL: @caller
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?x?x2xf32>, %[[ARG0_D1:.+]]: index, %[[ARG0_D2:.+]]: index,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<?xi32>, %[[ARG1_D0:.+]]: index)
func.func @caller(%arg0: tensor<4x?x?x2xf32>, %arg1: tensor<?xi32>) {
  // CHECK-NEXT: %[[TIED_ARG0:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<4x?x?x2xf32>{%[[ARG0_D1]], %[[ARG0_D2]]}
  // CHECK-NEXT: %[[TIED_ARG1:.+]] = flow.tensor.tie_shape %[[ARG1]] : tensor<?xi32>{%[[ARG1_D0]]}

  // CHECK: %[[RET:.+]]:5 = call @callee(%[[ARG0]], %[[ARG0_D1]], %[[ARG0_D2]], %[[ARG1]], %[[ARG1_D0]])
  // CHECK-SAME: (tensor<4x?x?x2xf32>, index, index, tensor<?xi32>, index) -> (tensor<4x?x?x2xf32>, index, index, tensor<?xi32>, index)
  %0:2 = call @callee(%arg0, %arg1) : (tensor<4x?x?x2xf32>, tensor<?xi32>) -> (tensor<4x?x?x2xf32>, tensor<?xi32>)

  // CHECK-NEXT: %[[TIED_RET0:.+]] = flow.tensor.tie_shape %[[RET]]#0 : tensor<4x?x?x2xf32>{%[[RET]]#1, %[[RET]]#2}
  // CHECK-NEXT: %[[TIED_RET1:.+]] = flow.tensor.tie_shape %[[RET]]#3 : tensor<?xi32>{%[[RET]]#4}

  // CHECK-NEXT: util.optimization_barrier %[[TIED_RET0]]
  util.optimization_barrier %0#0 : tensor<4x?x?x2xf32>
  // CHECK-NEXT: util.optimization_barrier %[[TIED_RET1]]
  util.optimization_barrier %0#1 : tensor<?xi32>

  return
}

func.func private @callee(%arg0: tensor<4x?x?x2xf32>, %arg1: tensor<?xi32>) -> (tensor<4x?x?x2xf32>, tensor<?xi32>)

// -----

// Tests that branch arguments are expanded.

// CHECK-LABEL: @br
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?x?x2xf32>, %[[ARG0_D1:.+]]: index, %[[ARG0_D2:.+]]: index,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<?xi32>, %[[ARG1_D0:.+]]: index)
func.func @br(%arg0: tensor<4x?x?x2xf32>, %arg1: tensor<?xi32>) {
  // CHECK-NEXT: %[[TIED_ARG0:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<4x?x?x2xf32>{%[[ARG0_D1]], %[[ARG0_D2]]}
  // CHECK-NEXT: %[[TIED_ARG1:.+]] = flow.tensor.tie_shape %[[ARG1]] : tensor<?xi32>{%[[ARG1_D0]]}

  // CHECK-NEXT: cf.br ^bb1(%[[ARG0]], %[[ARG0_D1]], %[[ARG0_D2]], %[[ARG1]], %[[ARG1_D0]]
  cf.br ^bb1(%arg0, %arg1 : tensor<4x?x?x2xf32>, tensor<?xi32>)

// CHECK-NEXT: ^bb1(%[[BB1_ARG0:.+]]: tensor<4x?x?x2xf32>, %[[BB1_ARG0_D1:.+]]: index, %[[BB1_ARG0_D2:.+]]: index,
// CHECK-SAME:      %[[BB1_ARG1:.+]]: tensor<?xi32>, %[[BB1_ARG1_D0:.+]]: index)
^bb1(%bb1_arg0: tensor<4x?x?x2xf32>, %bb1_arg1: tensor<?xi32>):
  // CHECK-NEXT: %[[TIED_BB1_ARG0:.+]] = flow.tensor.tie_shape %[[BB1_ARG0]] : tensor<4x?x?x2xf32>{%[[BB1_ARG0_D1]], %[[BB1_ARG0_D2]]}
  // CHECK-NEXT: %[[TIED_BB1_ARG1:.+]] = flow.tensor.tie_shape %[[BB1_ARG1]] : tensor<?xi32>{%[[BB1_ARG1_D0]]}

  // CHECK-NEXT: util.optimization_barrier %[[TIED_BB1_ARG0]]
  util.optimization_barrier %bb1_arg0 : tensor<4x?x?x2xf32>
  // CHECK-NEXT: util.optimization_barrier %[[TIED_BB1_ARG1]]
  util.optimization_barrier %bb1_arg1 : tensor<?xi32>

  return
}

// -----

// Tests that selects of dynamically shaped tensors expand to selecting dims.

// CHECK-LABEL: @select
// CHECK-SAME: (%[[COND:.+]]: i1,
// CHECK-SAME:  %[[ARG0:.+]]: tensor<4x?x?x2xf32>, %[[ARG0_D1:.+]]: index, %[[ARG0_D2:.+]]: index, %[[ARG1:.+]]: tensor<4x?x?x2xf32>, %[[ARG1_D1:.+]]: index, %[[ARG1_D2:.+]]: index)
func.func @select(%cond: i1, %arg0: tensor<4x?x?x2xf32>, %arg1: tensor<4x?x?x2xf32>) {
  // CHECK-NEXT: %[[TIED_ARG0:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<4x?x?x2xf32>{%[[ARG0_D1]], %[[ARG0_D2]]}
  // CHECK-NEXT: %[[TIED_ARG1:.+]] = flow.tensor.tie_shape %[[ARG1]] : tensor<4x?x?x2xf32>{%[[ARG1_D1]], %[[ARG1_D2]]}

  // CHECK-NEXT: %[[SEL_TENSOR:.+]] = arith.select %[[COND]], %[[TIED_ARG0]], %[[TIED_ARG1]]
  // CHECK-NEXT: %[[SEL_D1:.+]] = arith.select %[[COND]], %[[ARG0_D1]], %[[ARG1_D1]]
  // CHECK-NEXT: %[[SEL_D2:.+]] = arith.select %[[COND]], %[[ARG0_D2]], %[[ARG1_D2]]
  // CHECK-NEXT: %[[SEL_TIED:.+]] = flow.tensor.tie_shape %[[SEL_TENSOR]] : tensor<4x?x?x2xf32>{%[[SEL_D1]], %[[SEL_D2]]}
  %0 = arith.select %cond, %arg0, %arg1 : tensor<4x?x?x2xf32>

  // CHECK-NEXT: util.optimization_barrier %[[SEL_TIED]]
  util.optimization_barrier %0 : tensor<4x?x?x2xf32>

  return
}

// -----

// CHECK-LABEL: @scf_while
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: i32
func.func @scf_while(%arg0 : tensor<?xf32>, %arg1 : i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  // CHECK:  %[[TIE:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<?xf32>{%[[ARG1]]}
  // CHECK:  %[[C0:.+]] = arith.constant 0
  // CHECK:  %[[C1:.+]] = arith.constant 1
  // CHECK:  %[[WHILE:.+]]:3 = scf.while (%[[ARG3:.+]] = %[[C0]], %[[ARG4:.+]] = %[[ARG0]], %[[ARG5:.+]] = %[[ARG1]])
  %0:2 = scf.while(%arg3 = %zero, %arg4 = %arg0) : (i32, tensor<?xf32>) -> (i32, tensor<?xf32>) {
    // CHECK:    %[[TIE:.+]] = flow.tensor.tie_shape %[[ARG4]] : tensor<?xf32>{%[[ARG5]]}
    // CHECK:    %[[CMP:.+]] = arith.cmpi slt, %[[ARG3]], %[[ARG2]]
    // CHECK:    scf.condition(%[[CMP]]) %[[ARG3]], %[[ARG4]], %[[ARG5]]
    %1 = arith.cmpi slt, %arg3, %arg1 : i32
    scf.condition(%1) %arg3, %arg4 :  i32, tensor<?xf32>
  } do {
   ^bb0(%arg3: i32, %arg4 : tensor<?xf32>) :
    // CHECK:  ^bb0(%[[ARG3:.+]]: i32, %[[ARG4:.+]]: tensor<?xf32>, %[[ARG5:.+]]: index):
    // CHECK:    %[[TIE:.+]] = flow.tensor.tie_shape %[[ARG4]] : tensor<?xf32>{%arg5}
    // CHECK:    %[[ADDI:.+]] = arith.addi %[[ARG3]], %[[C1]]
    // CHECK:    %[[ADDF:.+]] = arith.addf %[[TIE]], %[[TIE]]
    // CHECK:    %[[C0:.+]] = arith.constant 0
    // CHECK:    %[[DIM:.+]] = tensor.dim %[[ADDF]], %[[C0]]
    // CHECK:    scf.yield %arg3, %6, %dim : i32, tensor<?xf32>, index
    %1 = arith.addi %arg3, %one : i32
    %2 = arith.addf %arg4, %arg4 : tensor<?xf32>
    scf.yield %arg3, %2 :  i32, tensor<?xf32>
  }

  // CHECK:  %[[TIE:.+]] = flow.tensor.tie_shape %[[WHILE]]#1 : tensor<?xf32>{%[[WHILE]]#2}
  // CHECK:  %[[BARRIER:.+]] = util.optimization_barrier %[[TIE]]
  util.optimization_barrier %0#1 : tensor<?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @scf_if
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: i1
func.func @scf_if(%arg0 : tensor<?xf32>, %arg1 : i1) {
  // CHECK:   %[[TIE:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<?xf32>{%[[ARG1]]}
  // CHECK:   %[[IF:.+]]:2 = scf.if %[[ARG2]] -> (tensor<?xf32>, index) {
  %0 = scf.if %arg1 -> tensor<?xf32> {
    // CHECK:     scf.yield %[[ARG0]], %[[ARG1]] : tensor<?xf32>, index
    scf.yield %arg0 : tensor<?xf32>
  } else {
    // CHECK:     %[[ADD:.+]] = arith.addf %[[TIE]], %[[TIE]]
    // CHECK:     %[[C0:.+]] = arith.constant 0 : index
    // CHECK:     %[[DIM:.+]] = tensor.dim %[[ADD]], %[[C0]]
    // CHECK:     scf.yield %[[ADD]], %[[DIM]]
    %1 = arith.addf %arg0, %arg0 : tensor<?xf32>
    scf.yield %1 : tensor<?xf32>
  }

  // CHECK:   %[[TIE:.+]] = flow.tensor.tie_shape %[[IF]]#0 : tensor<?xf32>{%[[IF]]#1}
  // CHECK:   %[[BARRIER:.+]] = util.optimization_barrier %[[TIE]] : tensor<?xf32>
  util.optimization_barrier %0 : tensor<?xf32>
  return
}
