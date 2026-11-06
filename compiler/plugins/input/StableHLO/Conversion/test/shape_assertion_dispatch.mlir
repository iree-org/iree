// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx --compile-to=flow %s | FileCheck %s
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=vmvx %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu %s -o /dev/null

// The assertion inside the scatter region must survive dispatch formation.
// CHECK: flow.executable
// CHECK: util.status.check_ok {{.*}}"negative current"
func.func @scatter_effect(%operand: tensor<4xf32>, %updates: tensor<2xf32>) -> tensor<4xf32> {
  %indices = stablehlo.constant dense<1> : tensor<1xi32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<f32>, %update: tensor<f32>):
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %pred = stablehlo.compare GE, %current, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%pred) {has_side_effect = true, error_message = "negative current"} : (tensor<i1>) -> ()
    stablehlo.return %update : tensor<f32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 0>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<1xi32>, tensor<2xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// Torchax's Dim(min=3, max=64) checks run on the host, before the one dispatch
// with one workload.
// CHECK-LABEL: util.func public @bounded_batch
// CHECK-NOT: flow.tensor.load
// CHECK-COUNT-4: util.status.check_ok
// CHECK-NOT: flow.tensor.load
// CHECK: flow.dispatch @{{[A-Za-z0-9_]+}}::@{{[A-Za-z0-9_]+}}[%{{[a-z0-9_]+}}](
// CHECK-NOT: flow.dispatch
// CHECK: util.return
func.func @bounded_batch(%arg0: tensor<?x7xf32>, %arg1: tensor<?x7xf32>) -> tensor<?x7xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %minusThree = stablehlo.constant dense<-3> : tensor<i32>
  %minusOne = stablehlo.constant dense<-1> : tensor<i32>
  %sixtyFour = stablehlo.constant dense<64> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x7xf32>) -> tensor<i32>
  %positive = stablehlo.compare GE, %size, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %size) {error_message = "b >= 1", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %otherSize = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x7xf32>) -> tensor<i32>
  %same = stablehlo.compare EQ, %otherSize, %size, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same, %otherSize, %size) {error_message = "arg1 batch != b", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  %aboveMin = stablehlo.add %size, %minusThree : tensor<i32>
  %minOk = stablehlo.compare GE, %aboveMin, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%minOk, %aboveMin, %size) {error_message = "b >= 3", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  %negated = stablehlo.multiply %size, %minusOne : tensor<i32>
  %belowMax = stablehlo.add %negated, %sixtyFour : tensor<i32>
  %maxOk = stablehlo.compare GE, %belowMax, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%maxOk, %belowMax, %size) {error_message = "b <= 64", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  %scaled = stablehlo.add %arg0, %arg0 : tensor<?x7xf32>
  %sum = stablehlo.multiply %scaled, %arg1 : tensor<?x7xf32>
  return %sum : tensor<?x7xf32>
}

// JAX states that three arguments share one symbol only through assertions;
// the dispatch then takes that symbol as its one workload.
// CHECK-LABEL: util.func public @shared_symbol
// CHECK-COUNT-2: util.status.check_ok
// CHECK: flow.dispatch @{{[A-Za-z0-9_]+}}::@{{[A-Za-z0-9_]+}}[%{{[a-z0-9_]+}}](
// CHECK-NOT: flow.dispatch
// CHECK: util.return
func.func @shared_symbol(%arg0: tensor<?x5xf32>, %arg1: tensor<?x5xf32>, %arg2: tensor<?x5xf32>) -> tensor<?x5xf32> {
  %b0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x5xf32>) -> tensor<i32>
  %b1 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x5xf32>) -> tensor<i32>
  %b2 = stablehlo.get_dimension_size %arg2, dim = 0 : (tensor<?x5xf32>) -> tensor<i32>
  %eq1 = stablehlo.compare EQ, %b0, %b1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %eq2 = stablehlo.compare EQ, %b0, %b2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq1) {has_side_effect = true, error_message = "arg1 dim 0 != b"} : (tensor<i1>) -> ()
  stablehlo.custom_call @shape_assertion(%eq2) {has_side_effect = true, error_message = "arg2 dim 0 != b"} : (tensor<i1>) -> ()
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x5xf32>
  %1 = stablehlo.exponential %0 : tensor<?x5xf32>
  %2 = stablehlo.multiply %1, %arg2 : tensor<?x5xf32>
  return %2 : tensor<?x5xf32>
}

// A derived dimension, 2*s, checks its remainder and quotient through JAX's
// floor-division helpers; the checks still run on the host.
// CHECK-LABEL: util.func public @derived_dimension
// CHECK-NOT: flow.tensor.load
// CHECK-COUNT-2: util.status.check_ok
// CHECK-NOT: flow.tensor.load
// CHECK: util.return
func.func @derived_dimension(%arg0: tensor<3x?x5xf32>) -> tensor<3x?x5xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %two = stablehlo.constant dense<2> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?x5xf32>) -> tensor<i32>
  %qr:2 = call @divmod(%size, %two) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %even = stablehlo.compare EQ, %qr#1, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%even, %qr#1) {error_message = "remainder {0}", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %positive = stablehlo.compare GE, %qr#0, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %qr#0) {error_message = "s = {0} < 1", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<3x?x5xf32>
  return %sum : tensor<3x?x5xf32>
}
func.func private @divmod(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %quotient = stablehlo.divide %x, %y : tensor<i32>
  %xSign = stablehlo.sign %x : tensor<i32>
  %ySign = stablehlo.sign %y : tensor<i32>
  %signsDiffer = stablehlo.compare NE, %xSign, %ySign, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %remainder = stablehlo.remainder %x, %y : tensor<i32>
  %inexact = stablehlo.compare NE, %remainder, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %roundDown = stablehlo.and %signsDiffer, %inexact : tensor<i1>
  %lower = stablehlo.subtract %quotient, %one : tensor<i32>
  %floor = stablehlo.select %roundDown, %lower, %quotient : tensor<i1>, tensor<i32>
  %remainderNegative = stablehlo.compare LT, %remainder, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %yNegative = stablehlo.compare LT, %y, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %wrongSign = stablehlo.compare NE, %remainderNegative, %yNegative, UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %adjust = stablehlo.and %wrongSign, %inexact : tensor<i1>
  %shifted = stablehlo.add %remainder, %y : tensor<i32>
  %modulo = stablehlo.select %adjust, %shifted, %remainder : tensor<i1>, tensor<i32>
  return %floor, %modulo : tensor<i32>, tensor<i32>
}
