// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-util-simplify-global-accesses)' %s | FileCheck %s

util.global private @varA = dense<1> : tensor<2xi32>
util.global private @varB = dense<3> : tensor<2x4xi32>

// CHECK-LABEL: @constants()
util.func public @constants() {
  // CHECK-DAG: constant 10
  %w = arith.constant 10 : index
  // CHECK-DAG: %[[VAR_A:.+]] = util.global.load @varA : tensor<2xi32>
  // CHECK-DAG: %[[VAR_B:.+]] = util.global.load @varB : tensor<2x4xi32>
  %varA = util.global.load @varA : tensor<2xi32>
  // CHECK-NEXT: %[[T:.+]] = flow.dispatch @ex::@dispatch0{{.+}}(%[[VAR_A]])
  %d0 = flow.dispatch @ex::@dispatch0[%w](%varA) : (tensor<2xi32>) -> tensor<2xi32>
  %varB = util.global.load @varB : tensor<2x4xi32>
  // CHECK-NEXT: flow.dispatch @ex::@dispatch1{{.+}}(%[[T]], %[[VAR_B]])
  %d1 = flow.dispatch @ex::@dispatch1[%w](%d0, %varB) : (tensor<2xi32>, tensor<2x4xi32>) -> tensor<2xi32>
  util.return
}

// -----

util.global private @varA = 1 : i32
util.global private @varB = 2 : i32

// CHECK-LABEL: @constants_in_cfg
util.func public @constants_in_cfg(%start: i32, %bound: i32) -> i32 {
  // CHECK-NEXT: %[[VAR_B:.+]] = util.global.load @varB : i32
  // CHECK-NEXT: %[[VAR_A:.+]] = util.global.load @varA : i32
  // CHECK-NEXT: cf.br ^bb1
  cf.br ^bb1(%start : i32)
// CHECK: ^bb1(%[[BB1_ARG:.+]]: i32):
^bb1(%2: i32):
  %cmp = arith.cmpi slt, %2, %bound : i32
  cf.cond_br %cmp, ^bb2(%2 : i32), ^bb3(%2 : i32)
// CHECK: ^bb2(%[[BB2_ARG:.+]]: i32):
^bb2(%5: i32):
  %6 = util.global.load @varA : i32
  // CHECK-NEXT: = arith.addi %[[BB2_ARG]], %[[VAR_A]] : i32
  %7 = arith.addi %5, %6 : i32
  cf.br ^bb1(%7 : i32)
// CHECK: ^bb3(%[[BB3_ARG:.+]]: i32):
^bb3(%8: i32):
  %9 = util.global.load @varA : i32
  // CHECK-NEXT: %[[T0:.+]] = arith.muli %[[BB3_ARG]], %[[VAR_A]] : i32
  %10 = arith.muli %8, %9 : i32
  %11 = util.global.load @varB : i32
  // CHECK-NEXT: %[[T1:.+]] = arith.subi %[[T0]], %[[VAR_B]]
  %12 = arith.subi %10, %11 : i32
  // CHECK-NEXT: util.return %[[T1]]
  util.return %12 : i32
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>
util.global private @varB = dense<3> : tensor<2x4xi32>

// CHECK-LABEL: @mixed_mutability
util.func public @mixed_mutability() {
  // CHECK-DAG: %[[VAR_A:.+]] = util.global.load @varA : tensor<2xi32>
  // CHECK-DAG: %[[VAR_B:.+]] = util.global.load @varB : tensor<2x4xi32>
  // CHECK-NEXT: constant 10
  %w = arith.constant 10 : index
  %varA = util.global.load @varA : tensor<2xi32>
  // CHECK-NEXT: %[[T0:.+]] = flow.dispatch @ex::@dispatch0{{.+}}(%[[VAR_A]])
  %d0 = flow.dispatch @ex::@dispatch0[%w](%varA) : (tensor<2xi32>) -> tensor<2xi32>
  %varB = util.global.load @varB : tensor<2x4xi32>
  // CHECK-NEXT: %[[T1:.+]] = flow.dispatch @ex::@dispatch1{{.+}}(%[[T0]], %[[VAR_B]])
  %d1 = flow.dispatch @ex::@dispatch1[%w](%d0, %varB) : (tensor<2xi32>, tensor<2x4xi32>) -> tensor<2xi32>
  // CHECK-NEXT: util.global.store %[[T1]], @varA : tensor<2xi32>
  util.global.store %d1, @varA : tensor<2xi32>
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @raw
util.func public @raw() {
  // CHECK: %[[T:.+]] = util.global.load @varA {id = 0
  %varA_0 = util.global.load @varA {id = 0} : tensor<2xi32>
  util.global.store %varA_0, @varA {id = 0} : tensor<2xi32>
  %varA_1 = util.global.load @varA {id = 1} : tensor<2xi32>
  // CHECK-NEXT: util.global.store %[[T]], @varA {id = 1
  util.global.store %varA_1, @varA {id = 1} : tensor<2xi32>
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @rar
util.func public @rar() -> (tensor<2xi32>, tensor<2xi32>) {
  // CHECK: %[[T:.+]] = util.global.load @varA {id = 0
  %varA_0 = util.global.load @varA {id = 0} : tensor<2xi32>
  %varA_1 = util.global.load @varA {id = 1} : tensor<2xi32>
  // CHECK-NEXT: util.return %[[T]], %[[T]]
  util.return %varA_0, %varA_1 : tensor<2xi32>, tensor<2xi32>
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @waw
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2xi32>, %[[ARG1:.+]]: tensor<2xi32>)
util.func public @waw(%varA_0: tensor<2xi32>, %varA_1: tensor<2xi32>) {
  util.global.store %varA_0, @varA : tensor<2xi32>
  // CHECK-NEXT: util.global.store %[[ARG1]], @varA
  util.global.store %varA_1, @varA : tensor<2xi32>
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @side_effects(
util.func public @side_effects() {
  // CHECK-NEXT: %[[T0:.+]] = util.global.load @varA
  %varA_0 = util.global.load @varA : tensor<2xi32>
  // CHECK-NEXT: util.global.store %[[T0]], @varA
  util.global.store %varA_0, @varA : tensor<2xi32>
  // CHECK-NEXT: util.call @other_fn()
  util.call @other_fn() : () -> ()
  // CHECK-NEXT: %[[T1:.+]] = util.global.load @varA
  %varA_1 = util.global.load @varA : tensor<2xi32>
  // CHECK-NEXT: util.global.store %[[T1]], @varA
  util.global.store %varA_1, @varA : tensor<2xi32>
  util.return
}

util.func private @other_fn()

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>
util.global private mutable @varB = dense<2> : tensor<2xi32>

// CHECK-LABEL: @ordering
util.func public @ordering() {
  %cst_top = arith.constant 1 : index
  %varA_0 = util.global.load @varA {id = 0} : tensor<2xi32>
  util.global.store %varA_0, @varA {id = 0} : tensor<2xi32>
  %varB_0 = util.global.load @varB {id = 1} : tensor<2xi32>
  util.global.store %varB_0, @varB {id = 1} : tensor<2xi32>
  %cst_bottom = arith.constant 2 : index

  // Loads should be moved up (in any order).
  // CHECK-DAG: %[[T0:.+]] = util.global.load @varA {id = 0
  // CHECK-DAG: %[[T1:.+]] = util.global.load @varB {id = 1
  // CHECK-NEXT: arith.constant

  // CHECK-NOT: NOT

  // Stores should be moved down (in any order).
  // CHECK-NEXT: arith.constant
  // CHECK-DAG: util.global.store %[[T0]], @varA {id = 0
  // CHECK-DAG: util.global.store %[[T1]], @varB {id = 1
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>
util.global private mutable @varB = dense<2> : tensor<2xi32>

util.func private @copy_global_b() {
  %varB_0 = util.global.load @varB : tensor<2xi32>
  util.global.store %varB_0, @varB : tensor<2xi32>
  util.return
}
util.func private @copy_global_a() {
  %varA_0 = util.global.load @varA : tensor<2xi32>
  util.global.store %varA_0, @varA : tensor<2xi32>
  util.return
}
util.func private @external()

// CHECK-LABEL: @blocking_scf
util.func public @blocking_scf(%steps: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %steps step %c1 {
    util.call @copy_global_a() : () -> ()
    scf.yield
  }

  %varA_1 = util.global.load @varA : tensor<2xi32>
  util.global.store %varA_1, @varA : tensor<2xi32>

  scf.for %i = %c0 to %steps step %c1 {
    util.call @copy_global_b() : () -> ()
    scf.yield
  }

  scf.for %i = %c0 to %steps step %c1 {
    util.call @external() : () -> ()
    scf.yield
  }

  // The first loop accesses @varA and blocks motion.
  //      CHECK: scf.for
  // CHECK-NEXT: util.call @copy_global_a
  // CHECK: %[[T0:.+]] = util.global.load @varA

  // The second loop only accesses @varB and does not block motion.
  //      CHECK: scf.for
  // CHECK-NEXT: util.call @copy_global_b
  //      CHECK: util.global.store %[[T0]], @varA

  // External calls are always blocking.
  //      CHECK: scf.for
  // CHECK-NEXT: util.call @external
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @blocking_indirect
util.func public @blocking_indirect(%arg0: !util.ptr<tensor<2xi32>>) {
  // CHECK-SAME: %[[PTR:.+]]: !util.ptr<tensor<2xi32>>
  %w = arith.constant 10 : index
  %varA = util.global.load @varA : tensor<2xi32>
  util.global.store %varA, @varA : tensor<2xi32>
  %d0 = flow.dispatch @ex::@dispatch0[%w](%varA) : (tensor<2xi32>) -> tensor<2xi32>
  util.global.store.indirect %d0, %arg0 : tensor<2xi32> -> !util.ptr<tensor<2xi32>>

  //  CHECK-DAG: %[[W:.+]] = arith.constant 10 : index
  //  CHECK-DAG: %[[LD:.+]] = util.global.load @varA : tensor<2xi32>
  // CHECK-NEXT: %[[D0:.+]] = flow.dispatch @ex::@dispatch0[%[[W]]](%[[LD]]) : (tensor<2xi32>) -> tensor<2xi32>

  // Propagate the store past the non-blocking dispatch, but not past the
  // indirect store.
  // CHECK-NEXT: util.global.store %[[LD]], @varA : tensor<2xi32>
  // CHECK-NEXT: util.global.store.indirect %[[D0]], %[[PTR]] : tensor<2xi32> -> !util.ptr<tensor<2xi32>>
  // CHECK-NEXT: util.return
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @hoisting_scf
util.func public @hoisting_scf(%steps: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %steps step %c1 {
    %varA_0 = util.global.load @varA : tensor<2xi32>
    util.global.store %varA_0, @varA : tensor<2xi32>
    scf.yield
  }
  %varA_1 = util.global.load @varA : tensor<2xi32>
  util.global.store %varA_1, @varA : tensor<2xi32>

  // CHECK: %[[T0:.+]] = util.global.load @varA
  //      CHECK: %[[LOOP_RES:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[T0]])
  // CHECK-NEXT:   scf.yield %[[ITER]] : tensor<2xi32>
  //      CHECK: util.global.store %[[LOOP_RES]], @varA
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @hoist_invariant_scf
util.func public @hoist_invariant_scf(%steps: index, %start: tensor<2xi32>) {
  // CHECK-SAME: %[[START:[A-Za-z0-9]+]]: tensor<2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %steps step %c1 {
    util.global.store %start, @varA : tensor<2xi32>
    scf.yield
  }
  %res = scf.for %i = %c0 to %steps step %c1 iter_args(%arg2 = %start) -> tensor<2xi32> {
    %varA = util.global.load @varA : tensor<2xi32>
    scf.yield %varA : tensor<2xi32>
  }

  //  CHECK-NOT: util.global.load
  //      CHECK: scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[START]])
  // CHECK-NEXT:   scf.yield %[[START]] : tensor<2xi32>
  // CHECK-NEXT: }
  // CHECK-NEXT: util.global.store %[[START]], @varA
  // CHECK-NEXT: util.return
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

util.func private @external()

// CHECK-LABEL: @hoist_blocking_external
util.func public @hoist_blocking_external(%steps: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %steps step %c1 {
    %varA_0 = util.global.load @varA : tensor<2xi32>
    util.call @external() : () -> ()
    util.global.store %varA_0, @varA : tensor<2xi32>
    scf.yield
  }

  // The external call within the loop body prevents hoisting of nearby
  // direct accessors.
  //      CHECK: scf.for
  // CHECK-NEXT:   %[[T0:.+]] = util.global.load @varA
  // CHECK-NEXT:   util.call @external
  // CHECK-NEXT:   util.global.store %[[T0]], @varA
  // CHECK-NEXT: }
  // CHECK-NEXT: util.return
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @hoisting_affine
util.func public @hoisting_affine() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  affine.for %i = 0 to 10 {
    %varA_0 = util.global.load @varA : tensor<2xi32>
    util.global.store %varA_0, @varA : tensor<2xi32>
    affine.yield
  }

  // CHECK: %[[T0:.+]] = util.global.load @varA
  //      CHECK: %[[LOOP_RES:.+]] = affine.for {{.*}} iter_args(%[[ITER:.+]] = %[[T0]])
  // CHECK-NEXT:   affine.yield %[[ITER]] : tensor<2xi32>
  //      CHECK: util.global.store %[[LOOP_RES]], @varA
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>

// CHECK-LABEL: @simplify_then_hoist
util.func public @simplify_then_hoist(%steps: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %steps step %c1 {
    %varA_0 = util.global.load @varA : tensor<2xi32>
    %varA_1 = util.global.load @varA : tensor<2xi32>
    util.global.store %varA_0, @varA : tensor<2xi32>
    util.global.store %varA_1, @varA : tensor<2xi32>
    scf.yield
  }

  // Ensure that we simplify the inner loop before hoisting to avoid redundant
  // iterators.
  // CHECK: %[[T0:.+]] = util.global.load @varA
  //      CHECK: %[[LOOP_RES:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[T0]])
  // CHECK-NEXT:   scf.yield %[[ITER]] : tensor<2xi32>
  //      CHECK: util.global.store %[[LOOP_RES]], @varA
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>
util.func private @external()

// CHECK-LABEL: @nested_scf
util.func public @nested_scf(%steps: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %steps step %c1 {
    scf.for %j = %c0 to %steps step %c1 {
      scf.for %k = %c0 to %steps step %c1 {
        %varA_0 = util.global.load @varA : tensor<2xi32>
        util.global.store %varA_0, @varA : tensor<2xi32>
        scf.yield
      }
      scf.yield
    }
    util.call @external() : () -> ()
    scf.yield
  }

  //      CHECK: scf.for
  // CHECK-NEXT:   %[[T0:.+]] = util.global.load @varA
  // CHECK-NEXT:   %[[L1:.+]] = scf.for {{.*}} iter_args(%[[ITER_J:.+]] = %[[T0]])
  // CHECK-NEXT:     %[[L2:.+]] = scf.for {{.*}} iter_args(%[[ITER_K:.+]] = %[[ITER_J]])
  // CHECK-NEXT:       scf.yield %[[ITER_K]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     scf.yield %[[L2]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   util.global.store %[[L1]], @varA
  // CHECK-NEXT:   util.call @external
  // CHECK-NEXT: }
  // CHECK-NEXT: util.return
  util.return
}

// -----

util.global private mutable @varA = dense<1> : tensor<2xi32>
util.global private mutable @varB = dense<2> : tensor<2xi32>
util.global private mutable @varC : index

// CHECK-LABEL: @multi_hoist
util.func public @multi_hoist(%steps: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %steps step %c1 {
    %varA_0 = util.global.load @varA : tensor<2xi32>
    %varB_0 = util.global.load @varB : tensor<2xi32>
    util.global.store %varA_0, @varA : tensor<2xi32>
    util.global.store %i, @varC : index
    util.global.store %varB_0, @varB : tensor<2xi32>
    scf.yield
  }

  //  CHECK-DAG: %[[A0:.+]] = util.global.load @varA
  //  CHECK-DAG: %[[B0:.+]] = util.global.load @varB
  //      CHECK: %[[LOOP_RES:.+]]:2 = scf.for
  // CHECK-SAME:     iter_args(%[[ITER_A:.+]] = %[[A0]], %[[ITER_B:.+]] = %[[B0]])
  // CHECK-NEXT:   util.global.store {{.*}} @varC
  // CHECK-NEXT:   scf.yield %[[ITER_A]], %[[ITER_B]] : tensor<2xi32>, tensor<2xi32>
  //  CHECK-DAG: util.global.store %[[LOOP_RES]]#0, @varA
  //  CHECK-DAG: util.global.store %[[LOOP_RES]]#1, @varB
  util.return
}
