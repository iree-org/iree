// RUN: iree-opt --split-input-file \
// RUN:     --iree-util-verify-initialization-order \
// RUN:     --iree-util-combine-initializers \
// RUN:     --iree-util-verify-initialization-order \
// RUN:     %s | FileCheck %s

// Tests that multiple initializers are combined in their module order.

util.func private @extern() -> index

// CHECK: util.global private mutable @global0 : index
util.global private mutable @global0 : index
// CHECK-NEXT: util.global private @global1 : index
util.global private @global1 : index
// CHECK-NEXT: util.global private @global2 : index
util.global private @global2 : index
// CHECK-NEXT: util.initializer {
// CHECK-NEXT: %[[VALUE0:.+]] = util.call @extern()
// CHECK-NEXT: util.global.store %[[VALUE0]], @global0
// CHECK-NEXT: %[[VALUE1:.+]] = util.call @extern()
// CHECK-NEXT: util.global.store %[[VALUE1]], @global1
// CHECK-NEXT: %[[VALUE2:.+]] = util.call @extern()
// CHECK-NEXT: util.global.store %[[VALUE2]], @global2
// CHECK-NEXT: util.return
util.initializer {
  %value0 = util.call @extern() : () -> index
  util.global.store %value0, @global0 : index
  util.return
}
// CHECK-NOT: util.initializer
util.initializer {
  %value1 = util.call @extern() : () -> index
  util.global.store %value1, @global1 : index
  %value2 = util.call @extern() : () -> index
  util.global.store %value2, @global2 : index
  util.return
}

// CHECK-LABEL: @orderedCombining
util.func public @orderedCombining(%arg0: index) -> (index, index, index) {
  util.global.store %arg0, @global0 : index
  %value0 = util.global.load @global0 : index
  %value1 = util.global.load @global1 : index
  %value2 = util.global.load @global2 : index
  util.return %value0, %value1, %value2 : index, index, index
}

// -----

// Tests that initializers containing CFG ops are inlined into the new combined
// initializer properly.

// CHECK: util.global private mutable @globalA : index
util.global private mutable @globalA : index
// CHECK: util.global private @globalB : index
util.global private @globalB : index
// CHECK: util.initializer {
// CHECK: ^bb1:
// CHECK:   cf.br ^bb3
// CHECK: ^bb2:
// CHECK:   cf.br ^bb3
// CHECK: ^bb3:
// CHECK:   cf.br ^bb4
// CHECK: ^bb4:
// CHECK:   util.return
// CHECK: }
util.initializer {
  %cond = arith.constant 1 : i1
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c100 = arith.constant 100 : index
  util.global.store %c100, @globalA : index
  cf.br ^bb3
^bb2:
  %c200 = arith.constant 200 : index
  util.global.store %c200, @globalA : index
  cf.br ^bb3
^bb3:
  util.return
}
// CHECK-NOT: util.initializer
util.initializer {
  %c300 = arith.constant 300 : index
  util.global.store %c300, @globalB : index
  util.return
}

// -----

// Tests that globals with initial values are materialized and combined with
// initializers in the correct order.

util.func private @side_effect() -> index

// CHECK: util.global private mutable @globalWithInit : index
// CHECK-NOT: = 42
util.global private mutable @globalWithInit = 42 : index

// CHECK-NEXT: util.global private mutable @globalFromInit : index
util.global private mutable @globalFromInit : index

// CHECK-NEXT: util.global private @immutableWithInit : index
// CHECK-NOT: = 100
util.global private @immutableWithInit = 100 : index

// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[C42:.+]] = arith.constant 42 : index
// CHECK-NEXT:   util.global.store %[[C42]], @globalWithInit
// CHECK-NEXT:   %[[C100:.+]] = arith.constant 100 : index
// CHECK-NEXT:   util.global.store %[[C100]], @immutableWithInit
// CHECK-NEXT:   %[[SIDE:.+]] = util.call @side_effect()
// CHECK-NEXT:   util.global.store %[[SIDE]], @globalFromInit
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer {
  %side = util.call @side_effect() : () -> index
  util.global.store %side, @globalFromInit : index
  util.return
}

// CHECK-LABEL: @interleavedGlobalsAndInits
util.func public @interleavedGlobalsAndInits() -> (index, index, index) {
  %v1 = util.global.load @globalWithInit : index
  %v2 = util.global.load @globalFromInit : index
  %v3 = util.global.load @immutableWithInit : index
  util.return %v1, %v2, %v3 : index, index, index
}

// -----

// Tests complex interleaving of globals with initial values and initializers.
// This verifies that order dependencies are correctly preserved.

// CHECK: util.global private @A : i32
util.global private @A = 1 : i32

// CHECK-NEXT: util.global private mutable @B : i32
util.global private mutable @B : i32

// CHECK-NEXT: util.global private @C : i32
util.global private @C = 3 : i32

// CHECK-NEXT: util.global private mutable @D : i32
util.global private mutable @D : i32

// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[C1:.+]] = arith.constant 1 : i32
// CHECK-NEXT:   util.global.store %[[C1]], @A
// CHECK-NEXT:   %[[C3:.+]] = arith.constant 3 : i32
// CHECK-NEXT:   util.global.store %[[C3]], @C
// CHECK-NEXT:   %[[A:.+]] = util.global.load @A
// CHECK-NEXT:   %[[C10:.+]] = arith.constant 10 : i32
// CHECK-NEXT:   %[[B_VAL:.+]] = arith.addi %[[A]], %[[C10]]
// CHECK-NEXT:   util.global.store %[[B_VAL]], @B
// CHECK-NEXT:   %[[B:.+]] = util.global.load @B
// CHECK-NEXT:   %[[C:.+]] = util.global.load @C
// CHECK-NEXT:   %[[D_VAL:.+]] = arith.muli %[[B]], %[[C]]
// CHECK-NEXT:   util.global.store %[[D_VAL]], @D
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer {
  // B = A + 10 (expects A = 1)
  %a = util.global.load @A : i32
  %c10 = arith.constant 10 : i32
  %b_val = arith.addi %a, %c10 : i32
  util.global.store %b_val, @B : i32
  util.return
}
util.initializer {
  // D = B * C (expects B = 11, C = 3)
  %b = util.global.load @B : i32
  %c = util.global.load @C : i32
  %d_val = arith.muli %b, %c : i32
  util.global.store %d_val, @D : i32
  util.return
}

// CHECK-LABEL: @verifyOrderDependencies
util.func public @verifyOrderDependencies() -> (i32, i32, i32, i32) {
  %a = util.global.load @A : i32
  %b = util.global.load @B : i32
  %c = util.global.load @C : i32
  %d = util.global.load @D : i32
  util.return %a, %b, %c, %d : i32, i32, i32, i32
}

// -----

// Tests that tensor initial values are properly materialized when combined
// with initializers.

// CHECK: util.global private @tensor_global : tensor<2xi32>
util.global private @tensor_global = dense<[100, 200]> : tensor<2xi32>

// CHECK: util.global private mutable @mutable_tensor : tensor<3xf32>
util.global private mutable @mutable_tensor : tensor<3xf32>

// CHECK: util.initializer {
// CHECK-DAG:   %[[TENSOR1:.+]] = arith.constant dense<[100, 200]> : tensor<2xi32>
// CHECK-DAG:   util.global.store %[[TENSOR1]], @tensor_global
// CHECK-DAG:   %[[TENSOR2:.+]] = arith.constant dense<[1.{{.*}}, 2.{{.*}}, 3.{{.*}}]> : tensor<3xf32>
// CHECK-DAG:   util.global.store %[[TENSOR2]], @mutable_tensor
// CHECK:   util.return
// CHECK: }
util.initializer {
  %tensor2 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  util.global.store %tensor2, @mutable_tensor : tensor<3xf32>
  util.return
}

// CHECK-LABEL: @tensorGlobals
util.func public @tensorGlobals() -> (tensor<2xi32>, tensor<3xf32>) {
  %t1 = util.global.load @tensor_global : tensor<2xi32>
  %t2 = util.global.load @mutable_tensor : tensor<3xf32>
  util.return %t1, %t2 : tensor<2xi32>, tensor<3xf32>
}

// -----

// Tests that we handle the case with no initializers (only globals with
// initial values) - we should not create an initializer in this case.

// CHECK: util.global private @only_global1 = 42 : i32
util.global private @only_global1 = 42 : i32

// CHECK: util.global private @only_global2 = 100 : i32
util.global private @only_global2 = 100 : i32

// CHECK-NOT: util.initializer

// CHECK-LABEL: @onlyGlobalsWithInitialValues
util.func public @onlyGlobalsWithInitialValues() -> (i32, i32) {
  %v1 = util.global.load @only_global1 : i32
  %v2 = util.global.load @only_global2 : i32
  util.return %v1, %v2 : i32, i32
}

// -----

// Tests single initializer (no combination needed).

// CHECK: util.global private mutable @single : index
util.global private mutable @single : index

// CHECK: util.initializer
// CHECK-NOT: util.initializer
util.initializer {
  %c42 = arith.constant 42 : index
  util.global.store %c42, @single : index
  util.return
}

// -----

// Tests empty module (no initializers or globals with initial values).

// CHECK-LABEL: util.func public @emptyModule
// CHECK-NOT: util.initializer
util.func public @emptyModule() {
  util.return
}

// -----

// Tests empty initializers (just util.return) are handled correctly.

// CHECK: util.global private @emptyInitGlobal : i32
util.global private @emptyInitGlobal = 123 : i32
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[C123:.+]] = arith.constant 123 : i32
// CHECK-NEXT:   util.global.store %[[C123]], @emptyInitGlobal
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer {
  // Empty initializer - just returns.
  util.return
}
util.initializer {
  // Another empty initializer.
  util.return
}

// -----

// Tests initializers with side effects that must preserve order.

util.func private @print(%arg0: i32)
util.func private @get_value() -> i32

// CHECK: util.global private mutable @sideEffectGlobal : i32
util.global private mutable @sideEffectGlobal = 0 : i32
// CHECK-NEXT: util.initializer
util.initializer {
  %c1 = arith.constant 1 : i32
  // CHECK: util.call @print(%c1_i32)
  util.call @print(%c1) : (i32) -> ()
  util.return
}
util.initializer {
  %c2 = arith.constant 2 : i32
  // CHECK: util.call @print(%c2_i32)
  util.call @print(%c2) : (i32) -> ()
  %val = util.call @get_value() : () -> i32
  // CHECK-NEXT: %[[VAL:.+]] = util.call @get_value
  // CHECK-NEXT: util.global.store %[[VAL]], @sideEffectGlobal
  util.global.store %val, @sideEffectGlobal : i32
  util.return
}
// CHECK: util.return

// -----

// Tests initializers with scf operations (loops and conditionals).

// CHECK: util.global private mutable @loopGlobal : index
util.global private mutable @loopGlobal : index
// CHECK-NEXT: util.initializer
util.initializer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // CHECK: %[[SUM:.+]] = scf.for
  %sum = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> index {
    %new_acc = arith.addi %acc, %i : index
    scf.yield %new_acc : index
  }
  // CHECK: util.global.store %[[SUM]], @loopGlobal
  util.global.store %sum, @loopGlobal : index
  util.return
}
// CHECK: util.return

// -----

// Tests many initializers to ensure scalability.

// CHECK: util.global private mutable @many0 : i32
util.global private mutable @many0 = 0 : i32
// CHECK-NEXT: util.global private mutable @many1 : i32
util.global private mutable @many1 = 1 : i32
// CHECK-NEXT: util.global private mutable @many2 : i32
util.global private mutable @many2 = 2 : i32
// CHECK-NEXT: util.global private mutable @many3 : i32
util.global private mutable @many3 = 3 : i32
// CHECK-NEXT: util.global private mutable @many4 : i32
util.global private mutable @many4 = 4 : i32
// CHECK-NEXT: util.initializer
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i32
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i32
util.initializer {
  %c10 = arith.constant 10 : i32
  util.global.store %c10, @many0 : i32
  util.return
}
util.initializer {
  %c11 = arith.constant 11 : i32
  util.global.store %c11, @many1 : i32
  util.return
}
util.initializer {
  %c12 = arith.constant 12 : i32
  util.global.store %c12, @many2 : i32
  util.return
}
util.initializer {
  %c13 = arith.constant 13 : i32
  util.global.store %c13, @many3 : i32
  util.return
}
util.initializer {
  %c14 = arith.constant 14 : i32
  util.global.store %c14, @many4 : i32
  util.return
}
// CHECK: util.return

// -----

// Tests initializers with util.unreachable operations.

// CHECK: util.global private mutable @unreachableGlobal : i32
util.global private mutable @unreachableGlobal : i32
// CHECK-NEXT: util.initializer
util.initializer {
  %cond = arith.constant false
  // CHECK: scf.if %{{.+}} {
  scf.if %cond {
    // This should never execute.
    // CHECK: util.scf.unreachable "impossible"
    util.scf.unreachable "impossible"
  } else {
    %c42 = arith.constant 42 : i32
    // CHECK: util.global.store %{{.+}}, @unreachableGlobal
    util.global.store %c42, @unreachableGlobal : i32
  }
  util.return
}
// CHECK: util.return

// -----

// Tests transitive dependencies through function calls in initializers.

util.func private @modify_globals() {
  %c999 = arith.constant 999 : i32
  util.global.store %c999, @transitiveA : i32
  util.return
}

// CHECK: util.global private mutable @transitiveA : i32
util.global private mutable @transitiveA = 1 : i32
// CHECK-NEXT: util.global private mutable @transitiveB : i32
util.global private mutable @transitiveB : i32
// CHECK-NEXT: util.initializer
util.initializer {
  // First set transitiveA to 1.
  // CHECK: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-NEXT: util.global.store %[[C1]], @transitiveA
  // Then call function that modifies it.
  // CHECK-NEXT: util.call @modify_globals()
  util.call @modify_globals() : () -> ()
  util.return
}
util.initializer {
  // Now read the modified value.
  // CHECK-NEXT: %[[LOADED:.+]] = util.global.load @transitiveA
  %val = util.global.load @transitiveA : i32
  // CHECK-NEXT: util.global.store %[[LOADED]], @transitiveB
  util.global.store %val, @transitiveB : i32
  util.return
}
// CHECK: util.return

// -----

// Tests initializers that modify globals before reading them (read-after-write).

// CHECK: util.global private mutable @rawGlobal : i32
util.global private mutable @rawGlobal = 5 : i32
// CHECK-NEXT: util.global private mutable @rawResult : i32
util.global private mutable @rawResult : i32
// CHECK-NEXT: util.initializer
util.initializer {
  // CHECK: %[[C5:.+]] = arith.constant 5 : i32
  // CHECK-NEXT: util.global.store %[[C5]], @rawGlobal
  // Write a different value.
  %c10 = arith.constant 10 : i32
  // CHECK-NEXT: %[[C10:.+]] = arith.constant 10 : i32
  // CHECK-NEXT: util.global.store %[[C10]], @rawGlobal
  util.global.store %c10, @rawGlobal : i32
  // Read back the value we just wrote.
  // CHECK-NEXT: %[[READ:.+]] = util.global.load @rawGlobal
  %val = util.global.load @rawGlobal : i32
  %c2 = arith.constant 2 : i32
  // CHECK-NEXT: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-NEXT: %[[MUL:.+]] = arith.muli %[[READ]], %[[C2]]
  %result = arith.muli %val, %c2 : i32
  // CHECK-NEXT: util.global.store %[[MUL]], @rawResult
  util.global.store %result, @rawResult : i32
  util.return
}
// CHECK: util.return

// -----

// Tests initializers with nested function calls and complex dependencies.

util.func private @level1() -> i32 {
  %val = util.global.load @nestedA : i32
  %c1 = arith.constant 1 : i32
  %result = arith.addi %val, %c1 : i32
  util.return %result : i32
}

util.func private @level2() -> i32 {
  %val = util.call @level1() : () -> i32
  %c10 = arith.constant 10 : i32
  %result = arith.muli %val, %c10 : i32
  util.return %result : i32
}

// CHECK: util.global private @nestedA : i32
util.global private @nestedA = 5 : i32
// CHECK-NEXT: util.global private mutable @nestedB : i32
util.global private mutable @nestedB : i32
// CHECK-NEXT: util.initializer
util.initializer {
  // CHECK: %[[C5:.+]] = arith.constant 5 : i32
  // CHECK-NEXT: util.global.store %[[C5]], @nestedA
  // CHECK-NEXT: %[[RES:.+]] = util.call @level2()
  %result = util.call @level2() : () -> i32
  // CHECK-NEXT: util.global.store %[[RES]], @nestedB
  util.global.store %result, @nestedB : i32
  util.return
}
// CHECK: util.return

// -----

// Tests interaction with util.optimization_barrier operations.

// CHECK: util.global private mutable @barrierGlobal : i32
util.global private mutable @barrierGlobal : i32
// CHECK-NEXT: util.initializer
util.initializer {
  %c42 = arith.constant 42 : i32
  // CHECK: %[[C42:.+]] = arith.constant 42 : i32
  // CHECK-NEXT: %[[BARRIER:.+]] = util.optimization_barrier %[[C42]]
  %barrier_val = util.optimization_barrier %c42 : i32
  // CHECK-NEXT: util.global.store %[[BARRIER]], @barrierGlobal
  util.global.store %barrier_val, @barrierGlobal : i32
  util.return
}
// CHECK: util.return

// -----

// Tests a non-materializable global alone keeps its initial value.

// CHECK: util.global private @non_mat = #util.byte_pattern<42> : i32
util.global private @non_mat = #util.byte_pattern<42> : i32
// CHECK-NOT: util.initializer

// -----

// Tests a non-materializable global before an initializer is preserved.

// CHECK: util.global private @non_mat = #util.byte_pattern<1> : i32
util.global private @non_mat = #util.byte_pattern<1> : i32
// CHECK-NEXT: util.global private mutable @g0 : i32
util.global private mutable @g0 : i32
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[C10:.+]] = arith.constant 10 : i32
// CHECK-NEXT:   util.global.store %[[C10]], @g0
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer {
  %c10 = arith.constant 10 : i32
  util.global.store %c10, @g0 : i32
  util.return
}

// -----

// Tests that non-materialized globals are included in the batch to preserve
// ordering. Since @non_mat appears before the initializer in module order it
// gets pulled into the batch with @g0 even though it is not used in the
// initializer.

// CHECK: util.global private mutable @g0 : i32
util.global private mutable @g0 : i32
// CHECK-NEXT: util.global private @non_mat = #util.byte_pattern<2> : i32
util.global private @non_mat = #util.byte_pattern<2> : i32
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[C20:.+]] = arith.constant 20 : i32
// CHECK-NEXT:   util.global.store %[[C20]], @g0
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer {
  %c20 = arith.constant 20 : i32
  util.global.store %c20, @g0 : i32
  util.return
}

// -----

// Tests an initializer + non-materializable + initializer should combine both.
// Non-materializable globals are included in the batch to demonstrate
// preservation of relative ordering.

// CHECK: util.global private mutable @g0 : i32
util.global private mutable @g0 : i32
// CHECK-NEXT: util.global private @non_mat = #util.byte_pattern<3> : i32
util.global private @non_mat = #util.byte_pattern<3> : i32
// CHECK-NEXT: util.global private mutable @g1 : i32
util.global private mutable @g1 : i32
// CHECK-NEXT: util.initializer {
// CHECK-DAG:   %[[C30:.+]] = arith.constant 30 : i32
// CHECK-DAG:   util.global.store %[[C30]], @g0
// CHECK-DAG:   %[[C40:.+]] = arith.constant 40 : i32
// CHECK-DAG:   util.global.store %[[C40]], @g1
// CHECK:   util.return
// CHECK-NEXT: }
util.initializer {
  %c30 = arith.constant 30 : i32
  util.global.store %c30, @g0 : i32
  util.return
}
// CHECK-NOT: util.initializer
util.initializer {
  %c40 = arith.constant 40 : i32
  util.global.store %c40, @g1 : i32
  util.return
}

// -----

// Tests that non-materializable globals maintain their relative position
// when combined with materializable globals.

// CHECK: util.global private @g0 : i32
// CHECK-NOT: = 5
util.global private @g0 = 5 : i32
// CHECK-NEXT: util.global private @non_mat = #util.byte_pattern<4> : i32
util.global private @non_mat = #util.byte_pattern<4> : i32
// CHECK-NEXT: util.global private @g1 : i32
// CHECK-NOT: = 6
util.global private @g1 = 6 : i32
// CHECK-NEXT: util.global private mutable @g2 : i32
util.global private mutable @g2 : i32
// CHECK-NEXT: util.initializer {
// CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : i32
// CHECK-DAG:   util.global.store %[[C5]], @g0
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : i32
// CHECK-DAG:   util.global.store %[[C6]], @g1
// CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : i32
// CHECK-DAG:   util.global.store %[[C7]], @g2
// CHECK:   util.return
// CHECK-NEXT: }
util.initializer {
  %c7 = arith.constant 7 : i32
  util.global.store %c7, @g2 : i32
  util.return
}

// -----

// Tests multiple non-materializable globals with no initializer stay as-is.

// CHECK: util.global private @non_mat1 = #util.byte_pattern<5> : i32
util.global private @non_mat1 = #util.byte_pattern<5> : i32
// CHECK-NEXT: util.global private @g0 = 8 : i32
util.global private @g0 = 8 : i32
// CHECK-NEXT: util.global private @non_mat2 = #util.byte_pattern<6> : i32
util.global private @non_mat2 = #util.byte_pattern<6> : i32
// CHECK-NEXT: util.global private @g1 = 9 : i32
util.global private @g1 = 9 : i32
// CHECK-NOT: util.initializer
