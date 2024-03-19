// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-util-simplify-global-accesses))' %s | FileCheck %s

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
  // CHECK-NEXT: %[[VAR_A:.+]] = util.global.load @varA : i32
  // CHECK-NEXT: %[[VAR_B:.+]] = util.global.load @varB : i32
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
