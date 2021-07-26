// RUN: iree-opt -split-input-file -iree-flow-simplify-variable-accesses %s | IreeFileCheck %s

flow.variable @varA dense<1> : tensor<2xi32> attributes {sym_visibility = "private"}
flow.variable @varB dense<3> : tensor<2x4xi32> attributes {sym_visibility = "private"}

// CHECK-LABEL: @constants()
func @constants() {
  // CHECK-NEXT: %[[VAR_A:.+]] = flow.variable.load @varA : tensor<2xi32>
  // CHECK-NEXT: %[[VAR_B:.+]] = flow.variable.load @varB : tensor<2x4xi32>
  // CHECK-NEXT: constant 10
  %w = constant 10 : index
  %varA = flow.variable.load @varA : tensor<2xi32>
  // CHECK-NEXT: %[[T:.+]] = flow.dispatch @ex::@dispatch0{{.+}}(%[[VAR_A]])
  %d0 = flow.dispatch @ex::@dispatch0[%w](%varA) : (tensor<2xi32>) -> tensor<2xi32>
  %varB = flow.variable.load @varB : tensor<2x4xi32>
  // CHECK-NEXT: flow.dispatch @ex::@dispatch1{{.+}}(%[[T]], %[[VAR_B]])
  %d1 = flow.dispatch @ex::@dispatch1[%w](%d0, %varB) : (tensor<2xi32>, tensor<2x4xi32>) -> tensor<2xi32>
  return
}

// -----

flow.variable @varA 1 : i32 attributes {sym_visibility = "private"}
flow.variable @varB 2 : i32 attributes {sym_visibility = "private"}

// CHECK-LABEL: @constants_in_cfg
func @constants_in_cfg(%start: i32, %bound: i32) -> i32 {
  // CHECK-NEXT: %[[VAR_A:.+]] = flow.variable.load @varA : i32
  // CHECK-NEXT: %[[VAR_B:.+]] = flow.variable.load @varB : i32
  // CHECK-NEXT: br ^bb1
  br ^bb1(%start : i32)
// CHECK: ^bb1(%[[BB1_ARG:.+]]: i32):
^bb1(%2: i32):
  %cmp = cmpi slt, %2, %bound : i32
  cond_br %cmp, ^bb2(%2 : i32), ^bb3(%2 : i32)
// CHECK: ^bb2(%[[BB2_ARG:.+]]: i32):
^bb2(%5: i32):
  %6 = flow.variable.load @varA : i32
  // CHECK-NEXT: = addi %[[BB2_ARG]], %[[VAR_A]] : i32
  %7 = addi %5, %6 : i32
  br ^bb1(%7 : i32)
// CHECK: ^bb3(%[[BB3_ARG:.+]]: i32):
^bb3(%8: i32):
  %9 = flow.variable.load @varA : i32
  // CHECK-NEXT: %[[T0:.+]] = muli %[[BB3_ARG]], %[[VAR_A]] : i32
  %10 = muli %8, %9 : i32
  %11 = flow.variable.load @varB : i32
  // CHECK-NEXT: %[[T1:.+]] = subi %[[T0]], %[[VAR_B]]
  %12 = subi %10, %11 : i32
  // CHECK-NEXT: return %[[T1]]
  return %12 : i32
}

// -----

flow.variable @varA mutable dense<1> : tensor<2xi32> attributes {sym_visibility = "private"}
flow.variable @varB dense<3> : tensor<2x4xi32> attributes {sym_visibility = "private"}

// CHECK-LABEL: @mixed_mutability
func @mixed_mutability() {
  // CHECK-DAG: %[[VAR_A:.+]] = flow.variable.load @varA : tensor<2xi32>
  // CHECK-DAG: %[[VAR_B:.+]] = flow.variable.load @varB : tensor<2x4xi32>
  // CHECK-NEXT: constant 10
  %w = constant 10 : index
  %varA = flow.variable.load @varA : tensor<2xi32>
  // CHECK-NEXT: %[[T0:.+]] = flow.dispatch @ex::@dispatch0{{.+}}(%[[VAR_A]])
  %d0 = flow.dispatch @ex::@dispatch0[%w](%varA) : (tensor<2xi32>) -> tensor<2xi32>
  %varB = flow.variable.load @varB : tensor<2x4xi32>
  // CHECK-NEXT: %[[T1:.+]] = flow.dispatch @ex::@dispatch1{{.+}}(%[[T0]], %[[VAR_B]])
  %d1 = flow.dispatch @ex::@dispatch1[%w](%d0, %varB) : (tensor<2xi32>, tensor<2x4xi32>) -> tensor<2xi32>
  // CHECK-NEXT: flow.variable.store %[[T1]], @varA : tensor<2xi32>
  flow.variable.store %d1, @varA : tensor<2xi32>
  return
}

// -----

flow.variable @varA mutable dense<1> : tensor<2xi32> attributes {sym_visibility = "private"}

// CHECK-LABEL: @raw
func @raw() {
  // CHECK: %[[T:.+]] = flow.variable.load @varA {id = 0
  %varA_0 = flow.variable.load @varA {id = 0} : tensor<2xi32>
  flow.variable.store %varA_0, @varA {id = 0} : tensor<2xi32>
  %varA_1 = flow.variable.load @varA {id = 1} : tensor<2xi32>
  // CHECK-NEXT: flow.variable.store %[[T]], @varA {id = 1
  flow.variable.store %varA_1, @varA {id = 1} : tensor<2xi32>
  return
}

// -----

flow.variable @varA mutable dense<1> : tensor<2xi32> attributes {sym_visibility = "private"}

// CHECK-LABEL: @rar
func @rar() -> (tensor<2xi32>, tensor<2xi32>) {
  // CHECK: %[[T:.+]] = flow.variable.load @varA {id = 0
  %varA_0 = flow.variable.load @varA {id = 0} : tensor<2xi32>
  %varA_1 = flow.variable.load @varA {id = 1} : tensor<2xi32>
  // CHECK-NEXT: return %[[T]], %[[T]]
  return %varA_0, %varA_1 : tensor<2xi32>, tensor<2xi32>
}

// -----

flow.variable @varA mutable dense<1> : tensor<2xi32> attributes {sym_visibility = "private"}

// CHECK-LABEL: @waw
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2xi32>, %[[ARG1:.+]]: tensor<2xi32>)
func @waw(%varA_0: tensor<2xi32>, %varA_1: tensor<2xi32>) {
  flow.variable.store %varA_0, @varA : tensor<2xi32>
  // CHECK-NEXT: flow.variable.store %[[ARG1]], @varA
  flow.variable.store %varA_1, @varA : tensor<2xi32>
  return
}

// -----

flow.variable @varA mutable dense<1> : tensor<2xi32> attributes {sym_visibility = "private"}

// CHECK-LABEL: @side_effects(
func @side_effects() {
  // CHECK-NEXT: %[[T0:.+]] = flow.variable.load @varA
  %varA_0 = flow.variable.load @varA : tensor<2xi32>
  // CHECK-NEXT: flow.variable.store %[[T0]], @varA
  flow.variable.store %varA_0, @varA : tensor<2xi32>
  // CHECK-NEXT: call @other_fn()
  call @other_fn() : () -> ()
  // CHECK-NEXT: %[[T1:.+]] = flow.variable.load @varA
  %varA_1 = flow.variable.load @varA : tensor<2xi32>
  // CHECK-NEXT: flow.variable.store %[[T1]], @varA
  flow.variable.store %varA_1, @varA : tensor<2xi32>
  return
}

func private @other_fn()
