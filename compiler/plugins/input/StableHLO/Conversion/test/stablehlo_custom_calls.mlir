// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-custom-calls),cse)" \
// RUN:   %s | FileCheck %s --implicit-check-not=util.status.check_ok --implicit-check-not="stablehlo.custom_call @shape_assertion"

// CHECK-LABEL: @householder
func.func public @householder(%arg0: tensor<4x3xf32>, %arg1: tensor<2xf32>) -> (tensor<4x3xf32>) {
  // CHECK: linalg.generic
  // CHECK: scf.for
  // CHECK:   linalg.generic
  // CHECK:   stablehlo.dot_general
  %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<4x3xf32>, tensor<2xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}


// -----

// 2*size wraps in i32 for sizes of 2^30 and more, so the select is not
// decided and the guard gives no bound.
// CHECK-LABEL: func.func @evaluator_wrap
// CHECK-NOT: util.assume.int
// CHECK: return
func.func @evaluator_wrap(%arg0: tensor<2x?xi8>) -> tensor<2x?xi8> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %two = stablehlo.constant dense<2> : tensor<i32>
  %hundred = stablehlo.constant dense<100> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<2x?xi8>) -> tensor<i32>
  %doubled = stablehlo.multiply %size, %two : tensor<i32>
  %nonnegative = stablehlo.compare GE, %doubled, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %chosen = stablehlo.select %nonnegative, %size, %zero : tensor<i1>, tensor<i32>
  %ok = stablehlo.compare LE, %chosen, %hundred, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok, %chosen) {error_message = "{0} > 100", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<2x?xi8>
  return %sum : tensor<2x?xi8>
}


// -----

// A comparison of unsigned constants without a comparison type is unsigned:
// 255 > 1 holds, so the select picks 0 and the guard gives no bound. Reading
// the constants as signed would pick the size instead.
// CHECK-LABEL: func.func @evaluator_unsigned_compare
// CHECK-NOT: util.assume.int
// CHECK: return
func.func @evaluator_unsigned_compare(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %hundred = stablehlo.constant dense<100> : tensor<i32>
  %max = stablehlo.constant dense<255> : tensor<ui8>
  %one = stablehlo.constant dense<1> : tensor<ui8>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<2x?xf32>) -> tensor<i32>
  %greater = stablehlo.compare GT, %max, %one : (tensor<ui8>, tensor<ui8>) -> tensor<i1>
  %chosen = stablehlo.select %greater, %zero, %size : tensor<i1>, tensor<i32>
  %ok = stablehlo.compare LE, %chosen, %hundred, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok, %chosen) {error_message = "{0} > 100", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<2x?xf32>
  return %sum : tensor<2x?xf32>
}


// -----

// An i1 add is a logical or: p + p is p, so the select picks 0 and the guard
// gives no bound.
// CHECK-LABEL: func.func @evaluator_bool_add
// CHECK-NOT: util.assume.int
// CHECK: return
func.func @evaluator_bool_add(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %hundred = stablehlo.constant dense<100> : tensor<i32>
  %true = stablehlo.constant dense<true> : tensor<i1>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<2x?xf32>) -> tensor<i32>
  %nonnegative = stablehlo.compare GE, %size, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %twice = stablehlo.add %nonnegative, %nonnegative : tensor<i1>
  %holds = stablehlo.compare EQ, %twice, %true, UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %chosen = stablehlo.select %holds, %zero, %size : tensor<i1>, tensor<i32>
  %ok = stablehlo.compare LE, %chosen, %hundred, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok, %chosen) {error_message = "{0} > 100", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<2x?xf32>
  return %sum : tensor<2x?xf32>
}


// -----

// Strict bounds on an exact quotient by 2 scale to the size.
// CHECK-LABEL: func.func @quotient_strict_bounds
// CHECK: util.assume.int %{{.+}}<umin = 4, umax = 8, udiv = 2> : i32
func.func @quotient_strict_bounds(%arg0: tensor<3x?xf32>) -> tensor<3x?xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %two = stablehlo.constant dense<2> : tensor<i32>
  %five = stablehlo.constant dense<5> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?xf32>) -> tensor<i32>
  %rem = stablehlo.remainder %size, %two : tensor<i32>
  %even = stablehlo.compare EQ, %rem, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%even) {error_message = "odd", has_side_effect = true} : (tensor<i1>) -> ()
  %half = stablehlo.divide %size, %two : tensor<i32>
  %above = stablehlo.compare GT, %half, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%above) {error_message = "small", has_side_effect = true} : (tensor<i1>) -> ()
  %below = stablehlo.compare LT, %half, %five, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%below) {error_message = "large", has_side_effect = true} : (tensor<i1>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<3x?xf32>
  return %sum : tensor<3x?xf32>
}


// -----

// A negated quotient bounds the size from above.
// CHECK-LABEL: func.func @quotient_negative_scale
// CHECK: util.assume.int %{{.+}}<umax = 8, udiv = 2> : i32
func.func @quotient_negative_scale(%arg0: tensor<3x?xf32>) -> tensor<3x?xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %two = stablehlo.constant dense<2> : tensor<i32>
  %six = stablehlo.constant dense<6> : tensor<i32>
  %ten = stablehlo.constant dense<10> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?xf32>) -> tensor<i32>
  %rem = stablehlo.remainder %size, %two : tensor<i32>
  %even = stablehlo.compare EQ, %rem, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%even) {error_message = "odd", has_side_effect = true} : (tensor<i1>) -> ()
  %half = stablehlo.divide %size, %two : tensor<i32>
  %left = stablehlo.subtract %ten, %half : tensor<i32>
  %ok = stablehlo.compare GE, %left, %six, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "large", has_side_effect = true} : (tensor<i1>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<3x?xf32>
  return %sum : tensor<3x?xf32>
}


// -----

// JAX with 64-bit types widens the size to i64 before bounding it; later uses
// of the widened size read the bounded one.
// CHECK-LABEL: func.func @widened_size_bounds
// CHECK: %[[BOUNDED:.+]] = util.assume.int %{{.+}}<umin = 1, umax = 8> : i32
// CHECK: %[[SIZE:.+]] = tensor.from_elements %[[BOUNDED]] : tensor<i32>
// CHECK: %[[WIDE:.+]] = stablehlo.convert %[[SIZE]] : (tensor<i32>) -> tensor<i64>
// CHECK: stablehlo.add %[[WIDE]], %[[WIDE]] : tensor<i64>
func.func @widened_size_bounds(%arg0: tensor<3x?xf32>) -> (tensor<3x?xf32>, tensor<i64>) {
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %minus_one = stablehlo.constant dense<-1> : tensor<i64>
  %eight = stablehlo.constant dense<8> : tensor<i64>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?xf32>) -> tensor<i32>
  %wide = stablehlo.convert %size : (tensor<i32>) -> tensor<i64>
  %less = stablehlo.add %wide, %minus_one : tensor<i64>
  %positive = stablehlo.compare GE, %less, %zero, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %less) {error_message = "{0} < 0", has_side_effect = true} : (tensor<i1>, tensor<i64>) -> ()
  %negated = stablehlo.multiply %wide, %minus_one : tensor<i64>
  %room = stablehlo.add %negated, %eight : tensor<i64>
  %fits = stablehlo.compare GE, %room, %zero, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%fits, %room) {error_message = "{0} < 0", has_side_effect = true} : (tensor<i1>, tensor<i64>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<3x?xf32>
  %twice = stablehlo.add %wide, %wide : tensor<i64>
  return %sum, %twice : tensor<3x?xf32>, tensor<i64>
}


// -----

// A value widened after i32 arithmetic may have wrapped, so it bounds nothing.
// CHECK-LABEL: func.func @widened_after_i32_arithmetic
// CHECK-NOT: util.assume.int
func.func @widened_after_i32_arithmetic(%arg0: tensor<3x?xf32>) -> tensor<3x?xf32> {
  %big = stablehlo.constant dense<1000000> : tensor<i32>
  %limit = stablehlo.constant dense<8000000> : tensor<i64>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?xf32>) -> tensor<i32>
  %scaled = stablehlo.multiply %size, %big : tensor<i32>
  %wide = stablehlo.convert %scaled : (tensor<i32>) -> tensor<i64>
  %fits = stablehlo.compare LE, %wide, %limit, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%fits, %wide) {error_message = "{0} too large", has_side_effect = true} : (tensor<i1>, tensor<i64>) -> ()
  %sum = stablehlo.add %arg0, %arg0 : tensor<3x?xf32>
  return %sum : tensor<3x?xf32>
}


// -----

// Two widenings of one size after the assertion both read the bounded size.
// Rewriting the first widening's operand must not stop the second from being
// recognised as a widening of the same size, so both fold to one value.
// CHECK-LABEL: func.func @widened_after_assertion
// CHECK: %[[SIZE:.+]] = stablehlo.get_dimension_size
// CHECK: %[[BOUNDED:.+]] = util.assume.int %{{.+}}<umin = 1> : i32
// CHECK: %[[TENSOR:.+]] = tensor.from_elements %[[BOUNDED]] : tensor<i32>
// CHECK: %[[WIDE:.+]] = stablehlo.convert %[[TENSOR]] : (tensor<i32>) -> tensor<i64>
// CHECK: return %[[WIDE]], %[[WIDE]]
func.func @widened_after_assertion(%arg0: tensor<?xf32>) -> (tensor<i64>, tensor<i64>) {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %positive = stablehlo.compare GT, %size, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive) {error_message = "positive", has_side_effect = true} : (tensor<i1>) -> ()
  %wide0 = stablehlo.convert %size : (tensor<i32>) -> tensor<i64>
  %wide1 = stablehlo.convert %size : (tensor<i32>) -> tensor<i64>
  return %wide0, %wide1 : tensor<i64>, tensor<i64>
}


// -----

// Negating INT64_MIN while matching a subtraction is not a valid proof step.
// CHECK-LABEL: func.func @shape_subtract_minimum
// CHECK-NOT: util.assume.int
// CHECK-NOT: flow.tensor.tie_shape
func.func @shape_subtract_minimum(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %minimum = stablehlo.constant dense<-9223372036854775808> : tensor<i64>
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %wide = stablehlo.convert %size : (tensor<i32>) -> tensor<i64>
  %difference = stablehlo.subtract %wide, %minimum : tensor<i64>
  %ok = stablehlo.compare GE, %difference, %zero, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "minimum subtraction", has_side_effect = true} : (tensor<i1>) -> ()
  return %arg0 : tensor<?xf32>
}


// -----

// Strict-bound adjustment must not overflow before an unusable fact is dropped.
// CHECK-LABEL: func.func @shape_strict_maximum
// CHECK-NOT: util.assume.int
func.func @shape_strict_maximum(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %maximum = stablehlo.constant dense<9223372036854775807> : tensor<i64>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %wide = stablehlo.convert %size : (tensor<i32>) -> tensor<i64>
  %ok = stablehlo.compare GT, %wide, %maximum, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "strict maximum", has_side_effect = true} : (tensor<i1>) -> ()
  return %arg0 : tensor<?xf32>
}


// -----

// Neither reversing a minimum bound nor subtracting an offset may overflow.
// CHECK-LABEL: func.func @shape_minimum_bound
// CHECK-NOT: util.assume.int
func.func @shape_minimum_bound(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %minimum = stablehlo.constant dense<-9223372036854775808> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %wide = stablehlo.convert %size : (tensor<i32>) -> tensor<i64>
  %offset = stablehlo.add %wide, %one : tensor<i64>
  %ok = stablehlo.compare GT, %minimum, %offset, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "minimum bound", has_side_effect = true} : (tensor<i1>) -> ()
  return %arg0 : tensor<?xf32>
}


// -----

// Facts inside a branch cannot change size uses in its enclosing block.
// CHECK-LABEL: func.func @shape_assertion_nested_scope
// CHECK: %[[SIZE:.+]] = stablehlo.get_dimension_size
// CHECK: %[[BRANCH:.+]] = scf.if
// CHECK: util.assume.int %{{.+}}<umin = 1> : i32
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[SIZE]] : tensor<i32>
// CHECK: return %[[BRANCH]], %[[SIZE]]
func.func @shape_assertion_nested_scope(%arg0: tensor<?xf32>, %choose: i1) -> (tensor<i32>, tensor<i32>) {
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %branch = scf.if %choose -> tensor<i32> {
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %inside = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %positive = stablehlo.compare GT, %inside, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%positive) {error_message = "branch bound", has_side_effect = true} : (tensor<i1>) -> ()
    scf.yield %inside : tensor<i32>
  } else {
    scf.yield %size : tensor<i32>
  }
  return %branch, %size : tensor<i32>, tensor<i32>
}


// -----

// Contradictory ranges suppress the assumption without discarding equalities.
// CHECK-LABEL: func.func @shape_contradictory_range_with_equality
// CHECK-NOT: util.assume.int
// CHECK: flow.tensor.tie_shape
// CHECK: flow.tensor.tie_shape
// CHECK-NOT: util.assume.int
func.func @shape_contradictory_range_with_equality(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  %five = stablehlo.constant dense<5> : tensor<i32>
  %ten = stablehlo.constant dense<10> : tensor<i32>
  %a_size = stablehlo.get_dimension_size %a, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %b_size = stablehlo.get_dimension_size %b, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %equal = stablehlo.compare EQ, %a_size, %b_size, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%equal) {error_message = "equal", has_side_effect = true} : (tensor<i1>) -> ()
  %lower = stablehlo.compare GT, %a_size, %ten, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%lower) {error_message = "lower", has_side_effect = true} : (tensor<i1>) -> ()
  %upper = stablehlo.compare LT, %b_size, %five, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%upper) {error_message = "upper", has_side_effect = true} : (tensor<i1>) -> ()
  %sum = stablehlo.add %a, %b : tensor<?xf32>
  return %sum : tensor<?xf32>
}

// -----

// Dropping shape assertions leaves other effects and the original scatter intact.
// CHECK-LABEL: func.func @scatter_with_other_effect
// CHECK-NOT: linalg.fill
// CHECK: %[[RESULT:.*]] = "stablehlo.scatter"
// CHECK: stablehlo.custom_call @other_effect
// CHECK: stablehlo.return
// CHECK-NOT: stablehlo.reduce
// CHECK: return %[[RESULT]]
func.func @scatter_with_other_effect(%operand: tensor<4xf32>, %indices: tensor<2x1xi32>, %updates: tensor<2xf32>) -> tensor<4xf32> {
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%current: tensor<f32>, %update: tensor<f32>):
    stablehlo.custom_call @other_effect(%current) {has_side_effect = true} : (tensor<f32>) -> ()
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %pred = stablehlo.compare GE, %current, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%pred) {has_side_effect = true, error_message = "negative current"} : (tensor<i1>) -> ()
    stablehlo.return %update : tensor<f32>
  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: @shape_assertion
// CHECK-SAME: %[[PRED:.*]]: tensor<i1>
// CHECK-NOT: tensor.extract
// CHECK: return
// CHECK-NOT: stablehlo.custom_call
func.func @shape_assertion(%predicate: tensor<i1>, %dimension: tensor<i32>) {
  stablehlo.custom_call @shape_assertion(%predicate, %dimension) {
    error_message = "expected a positive dimension", has_side_effect = true
  } : (tensor<i1>, tensor<i32>) -> ()
  return
}

// -----

// A vector predicate supplies no scalar shape facts; like every shape
// assertion, it is dropped without a runtime check, as upstream StableHLO
// lowering does.
// CHECK-LABEL: @shape_assertion_vector
// CHECK: return
func.func @shape_assertion_vector(%predicate: tensor<4xi1>) {
  stablehlo.custom_call @shape_assertion(%predicate) {
    error_message = "expected a scalar predicate", has_side_effect = true
  } : (tensor<4xi1>) -> ()
  return
}

// -----

// Missing predicates and messages supply no facts and do not prevent dropping.
// CHECK-LABEL: @shape_assertion_missing_predicate
// CHECK: return
func.func @shape_assertion_missing_predicate() {
  stablehlo.custom_call @shape_assertion() {
    error_message = "missing predicate", has_side_effect = true
  } : () -> ()
  return
}

// -----

// CHECK-LABEL: @shape_assertion_missing_message
// CHECK: return
func.func @shape_assertion_missing_message(%predicate: tensor<i1>) {
  stablehlo.custom_call @shape_assertion(%predicate) {
    has_side_effect = true
  } : (tensor<i1>) -> ()
  return
}
