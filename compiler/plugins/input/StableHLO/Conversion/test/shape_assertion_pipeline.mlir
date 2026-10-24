// RUN: iree-compile --compile-to=input %s | FileCheck %s --check-prefix=INPUT
// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --output-format=vm-asm %s | FileCheck %s --check-prefix=VM
// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --output-format=vm-asm --iree-opt-strip-assertions=true %s | FileCheck %s --check-prefix=STRIP

// INPUT-LABEL: util.func public @check_shape
// INPUT-DAG: %[[STATUS_OK:.*]] = arith.constant 0 : i32
// INPUT-DAG: %[[STATUS_FAILED:.*]] = arith.constant 9 : i32
// INPUT-DAG: %[[PREDICATE:.*]] = tensor.extract %arg0[] : tensor<i1>
// INPUT: %[[STATUS:.*]] = arith.select %[[PREDICATE]], %[[STATUS_OK]], %[[STATUS_FAILED]] : i32
// INPUT: util.status.check_ok %[[STATUS]], "expected a positive dimension"
// VM: vm.fail %{{.+}}, "expected a positive dimension"
// STRIP: vm.module
// STRIP: vm.fail %{{.+}}, "expected a positive dimension"
func.func @check_shape(%predicate: tensor<i1>) {
  stablehlo.custom_call @shape_assertion(%predicate) {
    error_message = "expected a positive dimension", has_side_effect = true
  } : (tensor<i1>) -> ()
  return
}

// Direct EQ checks over get_dimension_size values can share one runtime dim.
// The original comparisons remain before their checks, and each tensor is tied
// once after the last assertion.
// INPUT-LABEL: util.func public @equal_argument_dims
// INPUT-COUNT-2: arith.cmpi eq
// INPUT-COUNT-2: util.status.check_ok
// INPUT: flow.tensor.tie_shape
// INPUT: flow.tensor.tie_shape
// INPUT: flow.tensor.tie_shape
func.func @equal_argument_dims(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %dim0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %dim1 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %dim2 = stablehlo.get_dimension_size %arg2, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %eq01 = stablehlo.compare EQ, %dim0, %dim1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %eq02 = stablehlo.compare EQ, %dim0, %dim2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq01) {has_side_effect = true, error_message = "arg0 and arg1 differ"} : (tensor<i1>) -> ()
  stablehlo.custom_call @shape_assertion(%eq02, %dim2) {has_side_effect = true, error_message = "arg0 and arg2 differ"} : (tensor<i1>, tensor<i32>) -> ()
  %sum01 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
  %sum = stablehlo.add %sum01, %arg2 : tensor<?xf32>
  return %sum : tensor<?xf32>
}

// Independent axes on the same tensor pair produce one combined tie per
// tensor with both canonical dims.
// INPUT-LABEL: util.func public @equal_two_axes
// INPUT: util.status.check_ok
// INPUT: util.status.check_ok
// INPUT-COUNT-2: flow.tensor.tie_shape
func.func @equal_two_axes(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %lhs0 = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %rhs0 = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %lhs1 = stablehlo.get_dimension_size %lhs, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %rhs1 = stablehlo.get_dimension_size %rhs, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %eq0 = stablehlo.compare EQ, %lhs0, %rhs0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %eq1 = stablehlo.compare EQ, %lhs1, %rhs1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq0) {has_side_effect = true, error_message = "axis 0 differs"} : (tensor<i1>) -> ()
  stablehlo.custom_call @shape_assertion(%eq1) {has_side_effect = true, error_message = "axis 1 differs"} : (tensor<i1>) -> ()
  %sum = stablehlo.add %lhs, %rhs : tensor<?x?xf32>
  return %sum : tensor<?x?xf32>
}

// A non-EQ assertion and an assertion in a non-dominating branch do not
// establish shape equality for following operations.
// INPUT-LABEL: util.func public @non_equal_predicate
// INPUT: util.status.check_ok
// INPUT-NOT: flow.tensor.tie_shape
func.func @non_equal_predicate(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
  %lhsDim = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %rhsDim = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %greaterEqual = stablehlo.compare GE, %lhsDim, %rhsDim : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%greaterEqual) {has_side_effect = true, error_message = "lhs is smaller"} : (tensor<i1>) -> ()
  %sum = stablehlo.add %lhs, %rhs : tensor<?xf32>
  return %sum : tensor<?xf32>
}

// INPUT-LABEL: util.func public @branch_local_assertion
// INPUT: scf.if
// INPUT: util.status.check_ok
// INPUT-NOT: flow.tensor.tie_shape
func.func @branch_local_assertion(%condition: i1, %lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
  %lhsDim = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %rhsDim = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %equal = stablehlo.compare EQ, %lhsDim, %rhsDim : (tensor<i32>, tensor<i32>) -> tensor<i1>
  scf.if %condition {
    stablehlo.custom_call @shape_assertion(%equal) {has_side_effect = true, error_message = "dims differ"} : (tensor<i1>) -> ()
  }
  %sum = stablehlo.add %lhs, %rhs : tensor<?xf32>
  return %sum : tensor<?xf32>
}

// The checked relation makes this dynamic broadcast non-expanding. Without
// the assertion, the runtime-select path remains.
// INPUT-LABEL: util.func public @broadcast_asserted
// INPUT: util.status.check_ok
// INPUT: flow.tensor.tie_shape
// INPUT: flow.tensor.tie_shape
// INPUT-NOT: arith.select
func.func @broadcast_asserted(%lhs: tensor<?x8xf32>, %rhs: tensor<?xf32>) -> tensor<?x8xf32> {
  %lhsDim = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?x8xf32>) -> tensor<i32>
  %rhsDim = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %equal = stablehlo.compare EQ, %lhsDim, %rhsDim : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%equal) {has_side_effect = true, error_message = "batch dims differ"} : (tensor<i1>) -> ()
  %eight = stablehlo.constant dense<8> : tensor<1xi32>
  %batch = stablehlo.reshape %lhsDim : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %batch, %eight, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %broadcast = stablehlo.dynamic_broadcast_in_dim %rhs, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x8xf32>
  %sum = stablehlo.add %lhs, %broadcast : tensor<?x8xf32>
  %expected = stablehlo.constant dense<[[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0], [208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0]]> : tensor<2x8xf32>
  %expectedDynamic = tensor.cast %expected : tensor<2x8xf32> to tensor<?x8xf32>
  check.expect_eq(%sum, %expectedDynamic) : tensor<?x8xf32>
  return %sum : tensor<?x8xf32>
}

// INPUT-LABEL: util.func public @broadcast_unasserted
// INPUT: arith.select
func.func @broadcast_unasserted(%lhs: tensor<?x8xf32>, %rhs: tensor<?xf32>) -> tensor<?x8xf32> {
  %lhsDim = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?x8xf32>) -> tensor<i32>
  %eight = stablehlo.constant dense<8> : tensor<1xi32>
  %batch = stablehlo.reshape %lhsDim : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %batch, %eight, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %broadcast = stablehlo.dynamic_broadcast_in_dim %rhs, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x8xf32>
  %sum = stablehlo.add %lhs, %broadcast : tensor<?x8xf32>
  return %sum : tensor<?x8xf32>
}

func.func @assertion_fails(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
  %lhsDim = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %rhsDim = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %equal = stablehlo.compare EQ, %lhsDim, %rhsDim : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%equal) {has_side_effect = true, error_message = "dimensions must match"} : (tensor<i1>) -> ()
  return %lhs : tensor<?xf32>
}

// Torchax's Dim(min=3, max=64), as JAX asserts it.
// INPUT-LABEL: util.func public @bounded_dim
// INPUT-COUNT-3: util.status.check_ok
// INPUT: %[[BOUNDED:.+]] = util.assume.int %{{.+}}<umin = 3, umax = 64> : i32
// INPUT: %[[DIM:.+]] = arith.index_cast %[[BOUNDED]] : i32 to index
// INPUT: flow.tensor.tie_shape %{{.+}} : tensor<?x7xf32>{%[[DIM]]}
func.func @bounded_dim(%arg0: tensor<?x7xf32>) -> tensor<?x7xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %minusThree = stablehlo.constant dense<-3> : tensor<i32>
  %minusOne = stablehlo.constant dense<-1> : tensor<i32>
  %sixtyFour = stablehlo.constant dense<64> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x7xf32>) -> tensor<i32>
  %positive = stablehlo.compare GE, %size, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %size) {error_message = "b >= 1", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %aboveMin = stablehlo.add %size, %minusThree : tensor<i32>
  %minOk = stablehlo.compare GE, %aboveMin, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%minOk, %aboveMin, %size) {error_message = "b >= 3", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  %negated = stablehlo.multiply %size, %minusOne : tensor<i32>
  %belowMax = stablehlo.add %negated, %sixtyFour : tensor<i32>
  %maxOk = stablehlo.compare GE, %belowMax, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%maxOk, %belowMax, %size) {error_message = "b <= 64", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  %doubled = stablehlo.add %arg0, %arg0 : tensor<?x7xf32>
  return %doubled : tensor<?x7xf32>
}

// An equality spreads the range to the other argument.
// INPUT-LABEL: util.func public @bounded_equal_dims
// INPUT: %[[BOUNDED:.+]] = util.assume.int %{{.+}}<umin = 4, umax = 9> : i32
// INPUT: %[[DIM:.+]] = arith.index_cast %[[BOUNDED]] : i32 to index
// INPUT-DAG: flow.tensor.tie_shape %{{.+}} : tensor<?x5xf32>{%[[DIM]]}
// INPUT-DAG: flow.tensor.tie_shape %{{.+}} : tensor<?x5xf32>{%[[DIM]]}
func.func @bounded_equal_dims(%lhs: tensor<?x5xf32>, %rhs: tensor<?x5xf32>) -> tensor<?x5xf32> {
  %four = stablehlo.constant dense<4> : tensor<i32>
  %ten = stablehlo.constant dense<10> : tensor<i32>
  %lhsSize = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?x5xf32>) -> tensor<i32>
  %rhsSize = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?x5xf32>) -> tensor<i32>
  %equal = stablehlo.compare EQ, %lhsSize, %rhsSize, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%equal) {error_message = "sizes differ", has_side_effect = true} : (tensor<i1>) -> ()
  %minOk = stablehlo.compare LE, %four, %rhsSize, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%minOk) {error_message = "rhs below 4", has_side_effect = true} : (tensor<i1>) -> ()
  %maxOk = stablehlo.compare LT, %lhsSize, %ten, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%maxOk) {error_message = "lhs above 9", has_side_effect = true} : (tensor<i1>) -> ()
  %sum = stablehlo.add %lhs, %rhs : tensor<?x5xf32>
  return %sum : tensor<?x5xf32>
}

// The size feeding a broadcast shape reads the bounded dimension.
// INPUT-LABEL: util.func public @bounded_size_use
// INPUT: %[[BOUNDED:.+]] = util.assume.int %{{.+}}<umin = 2, umax = 16> : i32
// INPUT: flow.tensor.tie_shape
// INPUT: %[[BATCH:.+]] = arith.index_cast %[[BOUNDED]] : i32 to index
// INPUT-NEXT: tensor.empty(%[[BATCH]]) : tensor<?x3xf32>
func.func @bounded_size_use(%arg0: tensor<?xf32>) -> tensor<?x3xf32> {
  %two = stablehlo.constant dense<2> : tensor<i32>
  %sixteen = stablehlo.constant dense<16> : tensor<i32>
  %three = stablehlo.constant dense<3> : tensor<1xi32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %minOk = stablehlo.compare GE, %size, %two, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%minOk) {error_message = "b >= 2", has_side_effect = true} : (tensor<i1>) -> ()
  %maxOk = stablehlo.compare LE, %size, %sixteen : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%maxOk) {error_message = "b <= 16", has_side_effect = true} : (tensor<i1>) -> ()
  %batch = stablehlo.reshape %size : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %batch, %three, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %broadcast = stablehlo.dynamic_broadcast_in_dim %arg0, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x3xf32>
  return %broadcast : tensor<?x3xf32>
}

// A bound on an equal pair keeps the broadcast non-expanding.
// INPUT-LABEL: util.func public @broadcast_asserted_bounded
// INPUT: util.assume.int %{{.+}}<umin = 2> : i32
// INPUT: flow.tensor.tie_shape
// INPUT: flow.tensor.tie_shape
// INPUT-NOT: arith.select
func.func @broadcast_asserted_bounded(%lhs: tensor<?x8xf32>, %rhs: tensor<?xf32>) -> tensor<?x8xf32> {
  %two = stablehlo.constant dense<2> : tensor<i32>
  %lhsDim = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?x8xf32>) -> tensor<i32>
  %rhsDim = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %equal = stablehlo.compare EQ, %lhsDim, %rhsDim : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%equal) {has_side_effect = true, error_message = "batch dims differ"} : (tensor<i1>) -> ()
  %minOk = stablehlo.compare GE, %lhsDim, %two, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%minOk) {has_side_effect = true, error_message = "b >= 2"} : (tensor<i1>) -> ()
  %eight = stablehlo.constant dense<8> : tensor<1xi32>
  %batch = stablehlo.reshape %lhsDim : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %batch, %eight, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %broadcast = stablehlo.dynamic_broadcast_in_dim %rhs, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x8xf32>
  %sum = stablehlo.add %lhs, %broadcast : tensor<?x8xf32>
  return %sum : tensor<?x8xf32>
}

// JAX's own check on a dimension variable gives the lower bound it states.
// INPUT-LABEL: util.func public @jax_dimension_variable
// INPUT: util.assume.int %{{.+}}<umin = 1> : i32
func.func @jax_dimension_variable(%arg0: tensor<?x6xf32>) -> tensor<?x6xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x6xf32>) -> tensor<i32>
  %positive = stablehlo.compare GE, %size, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %size) {error_message = "b >= 1", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %doubled = stablehlo.add %arg0, %arg0 : tensor<?x6xf32>
  return %doubled : tensor<?x6xf32>
}

// Predicates that state no range, or one that wrapping or the comparison
// type makes unsound, leave the dimension unbounded.
// INPUT-LABEL: util.func public @unbounded_predicates
// INPUT-COUNT-5: util.status.check_ok
// INPUT-NOT: util.assume.int
func.func @unbounded_predicates(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %five = stablehlo.constant dense<5> : tensor<i32>
  %eight = stablehlo.constant dense<8> : tensor<i32>
  %minusTwo = stablehlo.constant dense<-2> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf32>) -> tensor<i32>
  %nonNegative = stablehlo.compare GE, %size, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%nonNegative) {error_message = "b >= 0", has_side_effect = true} : (tensor<i1>) -> ()
  %shifted = stablehlo.add %size, %five : tensor<i32>
  %wraps = stablehlo.compare LE, %shifted, %eight, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%wraps) {error_message = "b + 5 <= 8", has_side_effect = true} : (tensor<i1>) -> ()
  %unsigned = stablehlo.compare LE, %size, %eight, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%unsigned) {error_message = "b <= 8 unsigned", has_side_effect = true} : (tensor<i1>) -> ()
  %notFive = stablehlo.compare NE, %size, %five, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%notFive) {error_message = "b != 5", has_side_effect = true} : (tensor<i1>) -> ()
  %belowZero = stablehlo.compare GT, %size, %minusTwo, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%belowZero) {error_message = "b > -2", has_side_effect = true} : (tensor<i1>) -> ()
  %doubled = stablehlo.add %arg0, %arg0 : tensor<?x4xf32>
  return %doubled : tensor<?x4xf32>
}

// Contradictory bounds always fail at run time and give no fact.
// INPUT-LABEL: util.func public @empty_range
// INPUT-COUNT-2: util.status.check_ok
// INPUT-NOT: util.assume.int
func.func @empty_range(%arg0: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %four = stablehlo.constant dense<4> : tensor<i32>
  %nine = stablehlo.constant dense<9> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
  %minOk = stablehlo.compare GE, %size, %nine, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%minOk) {error_message = "b >= 9", has_side_effect = true} : (tensor<i1>) -> ()
  %maxOk = stablehlo.compare LE, %size, %four, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%maxOk) {error_message = "b <= 4", has_side_effect = true} : (tensor<i1>) -> ()
  %doubled = stablehlo.add %arg0, %arg0 : tensor<?x3xf32>
  return %doubled : tensor<?x3xf32>
}

// INPUT-LABEL: util.func public @branch_local_bound
// INPUT: scf.if
// INPUT: util.status.check_ok
// INPUT-NOT: util.assume.int
func.func @branch_local_bound(%condition: i1, %arg0: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %three = stablehlo.constant dense<3> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  %minOk = stablehlo.compare GE, %size, %three, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  scf.if %condition {
    stablehlo.custom_call @shape_assertion(%minOk) {error_message = "b >= 3", has_side_effect = true} : (tensor<i1>) -> ()
  }
  %doubled = stablehlo.add %arg0, %arg0 : tensor<?x2xf32>
  return %doubled : tensor<?x2xf32>
}

// Torchax's Dim 2*b on %arg1, as JAX asserts it.
// INPUT-LABEL: util.func public @expression_dim
// INPUT: %[[TWO:.+]] = arith.constant 2 : i32
// INPUT-COUNT-2: util.status.check_ok
// INPUT: %[[BATCH:.+]] = util.assume.int %{{.+}}<umin = 1> : i32
// INPUT: %[[TWICE:.+]] = arith.muli %[[BATCH]], %[[TWO]] : i32
// INPUT: %[[EVEN:.+]] = util.assume.int %[[TWICE]]<udiv = 2> : i32
// INPUT: %[[DIM:.+]] = arith.index_cast %[[EVEN]] : i32 to index
// INPUT: flow.tensor.tie_shape %arg1 : tensor<?x7xf32>{%[[DIM]]}
func.func @expression_dim(%arg0: tensor<?x7xf32>, %arg1: tensor<?x7xf32>) -> tensor<?x7xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %two = stablehlo.constant dense<2> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x7xf32>) -> tensor<i32>
  %otherSize = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x7xf32>) -> tensor<i32>
  %positive = stablehlo.compare GE, %size, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %size) {error_message = "b >= 1", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %twice = stablehlo.multiply %size, %two : tensor<i32>
  %same = stablehlo.compare EQ, %otherSize, %twice, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same, %otherSize, %twice, %size) {error_message = "arg1 batch != 2*b", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  %concat = stablehlo.concatenate %arg0, %arg0, dim = 0 : (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<?x7xf32>
  %product = stablehlo.multiply %concat, %arg1 : tensor<?x7xf32>
  return %product : tensor<?x7xf32>
}

// INPUT-LABEL: util.func public @expression_offset
// INPUT: util.status.check_ok
// INPUT: %[[SHIFTED:.+]] = arith.addi %{{.+}}, %{{.+}} : i32
// INPUT-NOT: udiv
// INPUT: %[[DIM:.+]] = arith.index_cast %[[SHIFTED]] : i32 to index
// INPUT: flow.tensor.tie_shape %arg1 : tensor<?x4xf32>{%[[DIM]]}
func.func @expression_offset(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %three = stablehlo.constant dense<3> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf32>) -> tensor<i32>
  %otherSize = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x4xf32>) -> tensor<i32>
  %shifted = stablehlo.add %three, %size : tensor<i32>
  %same = stablehlo.compare EQ, %shifted, %otherSize : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same) {error_message = "arg1 batch != b + 3", has_side_effect = true} : (tensor<i1>) -> ()
  %doubled = stablehlo.add %arg1, %arg1 : tensor<?x4xf32>
  return %doubled : tensor<?x4xf32>
}

// 3 * b can wrap back onto a valid size, so it proves nothing.
// INPUT-LABEL: util.func public @expression_wraps
// INPUT: util.status.check_ok
// INPUT-NOT: flow.tensor.tie_shape
// INPUT: util.return
func.func @expression_wraps(%arg0: tensor<?x6xf32>, %arg1: tensor<?x6xf32>) -> tensor<?x6xf32> {
  %three = stablehlo.constant dense<3> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x6xf32>) -> tensor<i32>
  %otherSize = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x6xf32>) -> tensor<i32>
  %thrice = stablehlo.multiply %size, %three : tensor<i32>
  %same = stablehlo.compare EQ, %otherSize, %thrice : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same) {error_message = "arg1 batch != 3*b", has_side_effect = true} : (tensor<i1>) -> ()
  %doubled = stablehlo.add %arg1, %arg1 : tensor<?x6xf32>
  return %doubled : tensor<?x6xf32>
}

// Dimensions of operation results take part like those of arguments.
// INPUT-LABEL: util.func public @result_dims_equal
// INPUT: util.status.check_ok
// INPUT: %[[DIM:.+]] = arith.index_cast
// INPUT-DAG: flow.tensor.tie_shape %{{.+}} : tensor<?x3xf32>{%[[DIM]]}
// INPUT-DAG: flow.tensor.tie_shape %{{.+}} : tensor<?x3xf32>{%[[DIM]]}
func.func @result_dims_equal(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %lhs = stablehlo.exponential %arg0 : tensor<?x3xf32>
  %rhs = stablehlo.negate %arg1 : tensor<?x3xf32>
  %lhsSize = stablehlo.get_dimension_size %lhs, dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
  %rhsSize = stablehlo.get_dimension_size %rhs, dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
  %same = stablehlo.compare EQ, %lhsSize, %rhsSize : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same) {error_message = "results differ", has_side_effect = true} : (tensor<i1>) -> ()
  %sum = stablehlo.add %lhs, %rhs : tensor<?x3xf32>
  return %sum : tensor<?x3xf32>
}

// The size of a dot result is its lhs batch; the asserted bias matches it.
// INPUT-LABEL: util.func public @dot_result_broadcast
// INPUT: util.status.check_ok
// INPUT-NOT: arith.select
// INPUT: util.return
func.func @dot_result_broadcast(%x: tensor<?x64xf32>, %w: tensor<64x32xf32>, %bias: tensor<?xf32>) -> tensor<?x32xf32> {
  %xb = stablehlo.get_dimension_size %x, dim = 0 : (tensor<?x64xf32>) -> tensor<i32>
  %bb = stablehlo.get_dimension_size %bias, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %eq = stablehlo.compare EQ, %xb, %bb : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq) {has_side_effect = true, error_message = "bias batch"} : (tensor<i1>) -> ()
  %z = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0] : (tensor<?x64xf32>, tensor<64x32xf32>) -> tensor<?x32xf32>
  %zb = stablehlo.get_dimension_size %z, dim = 0 : (tensor<?x32xf32>) -> tensor<i32>
  %zb1 = stablehlo.reshape %zb : (tensor<i32>) -> tensor<1xi32>
  %c32 = stablehlo.constant dense<32> : tensor<1xi32>
  %shape = stablehlo.concatenate %zb1, %c32, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %bc = stablehlo.dynamic_broadcast_in_dim %bias, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x32xf32>
  %r = stablehlo.add %z, %bc : tensor<?x32xf32>
  return %r : tensor<?x32xf32>
}

// A transpose moves the asserted dimension to the front.
// INPUT-LABEL: util.func public @transpose_result_broadcast
// INPUT: util.status.check_ok
// INPUT-NOT: arith.select
// INPUT: util.return
func.func @transpose_result_broadcast(%x: tensor<?x?xf32>, %bias: tensor<?xf32>) -> tensor<?x?xf32> {
  %x1 = stablehlo.get_dimension_size %x, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %bb = stablehlo.get_dimension_size %bias, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %eq = stablehlo.compare EQ, %x1, %bb : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%eq) {has_side_effect = true, error_message = "bias width"} : (tensor<i1>) -> ()
  %t = stablehlo.transpose %x, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %t0 = stablehlo.get_dimension_size %t, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %t1 = stablehlo.get_dimension_size %t, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %v0 = stablehlo.reshape %t0 : (tensor<i32>) -> tensor<1xi32>
  %v1 = stablehlo.reshape %t1 : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %v0, %v1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %bc = stablehlo.dynamic_broadcast_in_dim %bias, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %r = stablehlo.add %t, %bc : tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// An equality checked in one branch says nothing after the branch.
// INPUT-LABEL: util.func public @branch_local_transpose_broadcast
// INPUT: scf.if
// INPUT: util.status.check_ok
// INPUT: arith.cmpi eq, %{{.+}}, %c1 : index
func.func @branch_local_transpose_broadcast(%condition: i1, %x: tensor<?x?xf32>, %bias: tensor<?xf32>) -> tensor<?x?xf32> {
  %x1 = stablehlo.get_dimension_size %x, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %bb = stablehlo.get_dimension_size %bias, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %eq = stablehlo.compare EQ, %x1, %bb : (tensor<i32>, tensor<i32>) -> tensor<i1>
  scf.if %condition {
    stablehlo.custom_call @shape_assertion(%eq) {has_side_effect = true, error_message = "bias width"} : (tensor<i1>) -> ()
  }
  %t = stablehlo.transpose %x, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %t0 = stablehlo.get_dimension_size %t, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %t1 = stablehlo.get_dimension_size %t, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %v0 = stablehlo.reshape %t0 : (tensor<i32>) -> tensor<1xi32>
  %v1 = stablehlo.reshape %t1 : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %v0, %v1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %bc = stablehlo.dynamic_broadcast_in_dim %bias, %shape, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %r = stablehlo.add %t, %bc : tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// 2 * b - 1 wraps in i32 for b = 2^30 on the way to a valid size; the tie
// computes in i32 without overflow flags and widens only the exact result.
// INPUT-LABEL: util.func public @expression_twice_minus_one
// INPUT: util.status.check_ok
// INPUT: %[[TWICE:.+]] = arith.muli %{{.+}}, %c2_i32 : i32
// INPUT: %[[DIM_I32:.+]] = arith.addi %[[TWICE]], %c-1_i32 : i32
// INPUT: %[[DIM:.+]] = arith.index_cast %[[DIM_I32]] : i32 to index
// INPUT: flow.tensor.tie_shape %arg1 : tensor<?x5xf32>{%[[DIM]]}
func.func @expression_twice_minus_one(%arg0: tensor<?x3xf32>, %arg1: tensor<?x5xf32>) -> tensor<?x5xf32> {
  %two = stablehlo.constant dense<2> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
  %otherSize = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x5xf32>) -> tensor<i32>
  %twice = stablehlo.multiply %size, %two : tensor<i32>
  %expected = stablehlo.subtract %twice, %one : tensor<i32>
  %same = stablehlo.compare EQ, %otherSize, %expected : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same) {error_message = "arg1 batch != 2*b - 1", has_side_effect = true} : (tensor<i1>) -> ()
  %doubled = stablehlo.add %arg1, %arg1 : tensor<?x5xf32>
  return %doubled : tensor<?x5xf32>
}

// A long chain of steps by constants stays within the iterative walk.
// INPUT-LABEL: util.func public @long_bound_chain
// INPUT: util.assume.int %{{.+}}<umin = 3> : i32
func.func @long_bound_chain(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf32>) -> tensor<i32>
  %s0 = stablehlo.add %size, %one : tensor<i32>
  %s1 = stablehlo.subtract %s0, %one : tensor<i32>
  %s2 = stablehlo.add %s1, %one : tensor<i32>
  %s3 = stablehlo.subtract %s2, %one : tensor<i32>
  %s4 = stablehlo.add %s3, %one : tensor<i32>
  %s5 = stablehlo.subtract %s4, %one : tensor<i32>
  %s6 = stablehlo.add %s5, %one : tensor<i32>
  %s7 = stablehlo.subtract %s6, %one : tensor<i32>
  %s8 = stablehlo.add %s7, %one : tensor<i32>
  %s9 = stablehlo.subtract %s8, %one : tensor<i32>
  %s10 = stablehlo.add %s9, %one : tensor<i32>
  %s11 = stablehlo.subtract %s10, %one : tensor<i32>
  %s12 = stablehlo.add %s11, %one : tensor<i32>
  %s13 = stablehlo.subtract %s12, %one : tensor<i32>
  %s14 = stablehlo.add %s13, %one : tensor<i32>
  %s15 = stablehlo.subtract %s14, %one : tensor<i32>
  %s16 = stablehlo.add %s15, %one : tensor<i32>
  %s17 = stablehlo.subtract %s16, %one : tensor<i32>
  %s18 = stablehlo.add %s17, %one : tensor<i32>
  %s19 = stablehlo.subtract %s18, %one : tensor<i32>
  %s20 = stablehlo.add %s19, %one : tensor<i32>
  %s21 = stablehlo.subtract %s20, %one : tensor<i32>
  %s22 = stablehlo.add %s21, %one : tensor<i32>
  %s23 = stablehlo.subtract %s22, %one : tensor<i32>
  %s24 = stablehlo.add %s23, %one : tensor<i32>
  %s25 = stablehlo.subtract %s24, %one : tensor<i32>
  %s26 = stablehlo.add %s25, %one : tensor<i32>
  %s27 = stablehlo.subtract %s26, %one : tensor<i32>
  %s28 = stablehlo.add %s27, %one : tensor<i32>
  %s29 = stablehlo.subtract %s28, %one : tensor<i32>
  %s30 = stablehlo.add %s29, %one : tensor<i32>
  %s31 = stablehlo.subtract %s30, %one : tensor<i32>
  %s32 = stablehlo.add %s31, %one : tensor<i32>
  %s33 = stablehlo.subtract %s32, %one : tensor<i32>
  %s34 = stablehlo.add %s33, %one : tensor<i32>
  %s35 = stablehlo.subtract %s34, %one : tensor<i32>
  %s36 = stablehlo.add %s35, %one : tensor<i32>
  %s37 = stablehlo.subtract %s36, %one : tensor<i32>
  %s38 = stablehlo.add %s37, %one : tensor<i32>
  %s39 = stablehlo.subtract %s38, %one : tensor<i32>
  %s40 = stablehlo.add %s39, %one : tensor<i32>
  %s41 = stablehlo.subtract %s40, %one : tensor<i32>
  %s42 = stablehlo.add %s41, %one : tensor<i32>
  %s43 = stablehlo.subtract %s42, %one : tensor<i32>
  %s44 = stablehlo.add %s43, %one : tensor<i32>
  %s45 = stablehlo.subtract %s44, %one : tensor<i32>
  %s46 = stablehlo.add %s45, %one : tensor<i32>
  %s47 = stablehlo.subtract %s46, %one : tensor<i32>
  %s48 = stablehlo.add %s47, %one : tensor<i32>
  %s49 = stablehlo.subtract %s48, %one : tensor<i32>
  %s50 = stablehlo.add %s49, %one : tensor<i32>
  %s51 = stablehlo.subtract %s50, %one : tensor<i32>
  %s52 = stablehlo.add %s51, %one : tensor<i32>
  %s53 = stablehlo.subtract %s52, %one : tensor<i32>
  %s54 = stablehlo.add %s53, %one : tensor<i32>
  %s55 = stablehlo.subtract %s54, %one : tensor<i32>
  %s56 = stablehlo.add %s55, %one : tensor<i32>
  %s57 = stablehlo.subtract %s56, %one : tensor<i32>
  %s58 = stablehlo.add %s57, %one : tensor<i32>
  %s59 = stablehlo.subtract %s58, %one : tensor<i32>
  %s60 = stablehlo.add %s59, %one : tensor<i32>
  %s61 = stablehlo.subtract %s60, %one : tensor<i32>
  %s62 = stablehlo.add %s61, %one : tensor<i32>
  %s63 = stablehlo.subtract %s62, %one : tensor<i32>
  %s64 = stablehlo.add %s63, %one : tensor<i32>
  %s65 = stablehlo.subtract %s64, %one : tensor<i32>
  %s66 = stablehlo.add %s65, %one : tensor<i32>
  %s67 = stablehlo.subtract %s66, %one : tensor<i32>
  %s68 = stablehlo.add %s67, %one : tensor<i32>
  %s69 = stablehlo.subtract %s68, %one : tensor<i32>
  %s70 = stablehlo.add %s69, %one : tensor<i32>
  %s71 = stablehlo.subtract %s70, %one : tensor<i32>
  %s72 = stablehlo.add %s71, %one : tensor<i32>
  %s73 = stablehlo.subtract %s72, %one : tensor<i32>
  %s74 = stablehlo.add %s73, %one : tensor<i32>
  %s75 = stablehlo.subtract %s74, %one : tensor<i32>
  %s76 = stablehlo.add %s75, %one : tensor<i32>
  %s77 = stablehlo.subtract %s76, %one : tensor<i32>
  %s78 = stablehlo.add %s77, %one : tensor<i32>
  %s79 = stablehlo.subtract %s78, %one : tensor<i32>
  %s80 = stablehlo.add %s79, %one : tensor<i32>
  %s81 = stablehlo.subtract %s80, %one : tensor<i32>
  %s82 = stablehlo.add %s81, %one : tensor<i32>
  %s83 = stablehlo.subtract %s82, %one : tensor<i32>
  %s84 = stablehlo.add %s83, %one : tensor<i32>
  %s85 = stablehlo.subtract %s84, %one : tensor<i32>
  %s86 = stablehlo.add %s85, %one : tensor<i32>
  %s87 = stablehlo.subtract %s86, %one : tensor<i32>
  %s88 = stablehlo.add %s87, %one : tensor<i32>
  %s89 = stablehlo.subtract %s88, %one : tensor<i32>
  %s90 = stablehlo.add %s89, %one : tensor<i32>
  %s91 = stablehlo.subtract %s90, %one : tensor<i32>
  %s92 = stablehlo.add %s91, %one : tensor<i32>
  %s93 = stablehlo.subtract %s92, %one : tensor<i32>
  %s94 = stablehlo.add %s93, %one : tensor<i32>
  %s95 = stablehlo.subtract %s94, %one : tensor<i32>
  %s96 = stablehlo.add %s95, %one : tensor<i32>
  %s97 = stablehlo.subtract %s96, %one : tensor<i32>
  %s98 = stablehlo.add %s97, %one : tensor<i32>
  %s99 = stablehlo.subtract %s98, %one : tensor<i32>
  %s100 = stablehlo.add %s99, %one : tensor<i32>
  %s101 = stablehlo.subtract %s100, %one : tensor<i32>
  %s102 = stablehlo.add %s101, %one : tensor<i32>
  %s103 = stablehlo.subtract %s102, %one : tensor<i32>
  %s104 = stablehlo.add %s103, %one : tensor<i32>
  %s105 = stablehlo.subtract %s104, %one : tensor<i32>
  %s106 = stablehlo.add %s105, %one : tensor<i32>
  %s107 = stablehlo.subtract %s106, %one : tensor<i32>
  %s108 = stablehlo.add %s107, %one : tensor<i32>
  %s109 = stablehlo.subtract %s108, %one : tensor<i32>
  %s110 = stablehlo.add %s109, %one : tensor<i32>
  %s111 = stablehlo.subtract %s110, %one : tensor<i32>
  %s112 = stablehlo.add %s111, %one : tensor<i32>
  %s113 = stablehlo.subtract %s112, %one : tensor<i32>
  %s114 = stablehlo.add %s113, %one : tensor<i32>
  %s115 = stablehlo.subtract %s114, %one : tensor<i32>
  %s116 = stablehlo.add %s115, %one : tensor<i32>
  %s117 = stablehlo.subtract %s116, %one : tensor<i32>
  %s118 = stablehlo.add %s117, %one : tensor<i32>
  %s119 = stablehlo.subtract %s118, %one : tensor<i32>
  %s120 = stablehlo.add %s119, %one : tensor<i32>
  %s121 = stablehlo.subtract %s120, %one : tensor<i32>
  %s122 = stablehlo.add %s121, %one : tensor<i32>
  %s123 = stablehlo.subtract %s122, %one : tensor<i32>
  %s124 = stablehlo.add %s123, %one : tensor<i32>
  %s125 = stablehlo.subtract %s124, %one : tensor<i32>
  %s126 = stablehlo.add %s125, %one : tensor<i32>
  %s127 = stablehlo.subtract %s126, %one : tensor<i32>
  %s128 = stablehlo.add %s127, %one : tensor<i32>
  %s129 = stablehlo.subtract %s128, %one : tensor<i32>
  %s130 = stablehlo.add %s129, %one : tensor<i32>
  %s131 = stablehlo.subtract %s130, %one : tensor<i32>
  %s132 = stablehlo.add %s131, %one : tensor<i32>
  %s133 = stablehlo.subtract %s132, %one : tensor<i32>
  %s134 = stablehlo.add %s133, %one : tensor<i32>
  %s135 = stablehlo.subtract %s134, %one : tensor<i32>
  %s136 = stablehlo.add %s135, %one : tensor<i32>
  %s137 = stablehlo.subtract %s136, %one : tensor<i32>
  %s138 = stablehlo.add %s137, %one : tensor<i32>
  %s139 = stablehlo.subtract %s138, %one : tensor<i32>
  %s140 = stablehlo.add %s139, %one : tensor<i32>
  %s141 = stablehlo.subtract %s140, %one : tensor<i32>
  %s142 = stablehlo.add %s141, %one : tensor<i32>
  %s143 = stablehlo.subtract %s142, %one : tensor<i32>
  %s144 = stablehlo.add %s143, %one : tensor<i32>
  %s145 = stablehlo.subtract %s144, %one : tensor<i32>
  %s146 = stablehlo.add %s145, %one : tensor<i32>
  %s147 = stablehlo.subtract %s146, %one : tensor<i32>
  %s148 = stablehlo.add %s147, %one : tensor<i32>
  %s149 = stablehlo.subtract %s148, %one : tensor<i32>
  %s150 = stablehlo.add %s149, %one : tensor<i32>
  %s151 = stablehlo.subtract %s150, %one : tensor<i32>
  %s152 = stablehlo.add %s151, %one : tensor<i32>
  %s153 = stablehlo.subtract %s152, %one : tensor<i32>
  %s154 = stablehlo.add %s153, %one : tensor<i32>
  %s155 = stablehlo.subtract %s154, %one : tensor<i32>
  %s156 = stablehlo.add %s155, %one : tensor<i32>
  %s157 = stablehlo.subtract %s156, %one : tensor<i32>
  %s158 = stablehlo.add %s157, %one : tensor<i32>
  %s159 = stablehlo.subtract %s158, %one : tensor<i32>
  %s160 = stablehlo.add %s159, %one : tensor<i32>
  %s161 = stablehlo.subtract %s160, %one : tensor<i32>
  %s162 = stablehlo.add %s161, %one : tensor<i32>
  %s163 = stablehlo.subtract %s162, %one : tensor<i32>
  %s164 = stablehlo.add %s163, %one : tensor<i32>
  %s165 = stablehlo.subtract %s164, %one : tensor<i32>
  %s166 = stablehlo.add %s165, %one : tensor<i32>
  %s167 = stablehlo.subtract %s166, %one : tensor<i32>
  %s168 = stablehlo.add %s167, %one : tensor<i32>
  %s169 = stablehlo.subtract %s168, %one : tensor<i32>
  %s170 = stablehlo.add %s169, %one : tensor<i32>
  %s171 = stablehlo.subtract %s170, %one : tensor<i32>
  %s172 = stablehlo.add %s171, %one : tensor<i32>
  %s173 = stablehlo.subtract %s172, %one : tensor<i32>
  %s174 = stablehlo.add %s173, %one : tensor<i32>
  %s175 = stablehlo.subtract %s174, %one : tensor<i32>
  %s176 = stablehlo.add %s175, %one : tensor<i32>
  %s177 = stablehlo.subtract %s176, %one : tensor<i32>
  %s178 = stablehlo.add %s177, %one : tensor<i32>
  %s179 = stablehlo.subtract %s178, %one : tensor<i32>
  %s180 = stablehlo.add %s179, %one : tensor<i32>
  %s181 = stablehlo.subtract %s180, %one : tensor<i32>
  %s182 = stablehlo.add %s181, %one : tensor<i32>
  %s183 = stablehlo.subtract %s182, %one : tensor<i32>
  %s184 = stablehlo.add %s183, %one : tensor<i32>
  %s185 = stablehlo.subtract %s184, %one : tensor<i32>
  %s186 = stablehlo.add %s185, %one : tensor<i32>
  %s187 = stablehlo.subtract %s186, %one : tensor<i32>
  %s188 = stablehlo.add %s187, %one : tensor<i32>
  %s189 = stablehlo.subtract %s188, %one : tensor<i32>
  %s190 = stablehlo.add %s189, %one : tensor<i32>
  %s191 = stablehlo.subtract %s190, %one : tensor<i32>
  %s192 = stablehlo.add %s191, %one : tensor<i32>
  %s193 = stablehlo.subtract %s192, %one : tensor<i32>
  %s194 = stablehlo.add %s193, %one : tensor<i32>
  %s195 = stablehlo.subtract %s194, %one : tensor<i32>
  %s196 = stablehlo.add %s195, %one : tensor<i32>
  %s197 = stablehlo.subtract %s196, %one : tensor<i32>
  %s198 = stablehlo.add %s197, %one : tensor<i32>
  %s199 = stablehlo.subtract %s198, %one : tensor<i32>
  %s200 = stablehlo.add %s199, %one : tensor<i32>
  %s201 = stablehlo.subtract %s200, %one : tensor<i32>
  %s202 = stablehlo.add %s201, %one : tensor<i32>
  %s203 = stablehlo.subtract %s202, %one : tensor<i32>
  %s204 = stablehlo.add %s203, %one : tensor<i32>
  %s205 = stablehlo.subtract %s204, %one : tensor<i32>
  %s206 = stablehlo.add %s205, %one : tensor<i32>
  %s207 = stablehlo.subtract %s206, %one : tensor<i32>
  %s208 = stablehlo.add %s207, %one : tensor<i32>
  %s209 = stablehlo.subtract %s208, %one : tensor<i32>
  %s210 = stablehlo.add %s209, %one : tensor<i32>
  %s211 = stablehlo.subtract %s210, %one : tensor<i32>
  %s212 = stablehlo.add %s211, %one : tensor<i32>
  %s213 = stablehlo.subtract %s212, %one : tensor<i32>
  %s214 = stablehlo.add %s213, %one : tensor<i32>
  %s215 = stablehlo.subtract %s214, %one : tensor<i32>
  %s216 = stablehlo.add %s215, %one : tensor<i32>
  %s217 = stablehlo.subtract %s216, %one : tensor<i32>
  %s218 = stablehlo.add %s217, %one : tensor<i32>
  %s219 = stablehlo.subtract %s218, %one : tensor<i32>
  %s220 = stablehlo.add %s219, %one : tensor<i32>
  %s221 = stablehlo.subtract %s220, %one : tensor<i32>
  %s222 = stablehlo.add %s221, %one : tensor<i32>
  %s223 = stablehlo.subtract %s222, %one : tensor<i32>
  %s224 = stablehlo.add %s223, %one : tensor<i32>
  %s225 = stablehlo.subtract %s224, %one : tensor<i32>
  %s226 = stablehlo.add %s225, %one : tensor<i32>
  %s227 = stablehlo.subtract %s226, %one : tensor<i32>
  %s228 = stablehlo.add %s227, %one : tensor<i32>
  %s229 = stablehlo.subtract %s228, %one : tensor<i32>
  %s230 = stablehlo.add %s229, %one : tensor<i32>
  %s231 = stablehlo.subtract %s230, %one : tensor<i32>
  %s232 = stablehlo.add %s231, %one : tensor<i32>
  %s233 = stablehlo.subtract %s232, %one : tensor<i32>
  %s234 = stablehlo.add %s233, %one : tensor<i32>
  %s235 = stablehlo.subtract %s234, %one : tensor<i32>
  %s236 = stablehlo.add %s235, %one : tensor<i32>
  %s237 = stablehlo.subtract %s236, %one : tensor<i32>
  %s238 = stablehlo.add %s237, %one : tensor<i32>
  %s239 = stablehlo.subtract %s238, %one : tensor<i32>
  %s240 = stablehlo.add %s239, %one : tensor<i32>
  %s241 = stablehlo.subtract %s240, %one : tensor<i32>
  %s242 = stablehlo.add %s241, %one : tensor<i32>
  %s243 = stablehlo.subtract %s242, %one : tensor<i32>
  %s244 = stablehlo.add %s243, %one : tensor<i32>
  %s245 = stablehlo.subtract %s244, %one : tensor<i32>
  %s246 = stablehlo.add %s245, %one : tensor<i32>
  %s247 = stablehlo.subtract %s246, %one : tensor<i32>
  %s248 = stablehlo.add %s247, %one : tensor<i32>
  %s249 = stablehlo.subtract %s248, %one : tensor<i32>
  %s250 = stablehlo.add %s249, %one : tensor<i32>
  %s251 = stablehlo.subtract %s250, %one : tensor<i32>
  %s252 = stablehlo.add %s251, %one : tensor<i32>
  %s253 = stablehlo.subtract %s252, %one : tensor<i32>
  %s254 = stablehlo.add %s253, %one : tensor<i32>
  %s255 = stablehlo.subtract %s254, %one : tensor<i32>
  %s256 = stablehlo.add %s255, %one : tensor<i32>
  %s257 = stablehlo.subtract %s256, %one : tensor<i32>
  %s258 = stablehlo.add %s257, %one : tensor<i32>
  %s259 = stablehlo.subtract %s258, %one : tensor<i32>
  %s260 = stablehlo.add %s259, %one : tensor<i32>
  %s261 = stablehlo.subtract %s260, %one : tensor<i32>
  %s262 = stablehlo.add %s261, %one : tensor<i32>
  %s263 = stablehlo.subtract %s262, %one : tensor<i32>
  %s264 = stablehlo.add %s263, %one : tensor<i32>
  %s265 = stablehlo.subtract %s264, %one : tensor<i32>
  %s266 = stablehlo.add %s265, %one : tensor<i32>
  %s267 = stablehlo.subtract %s266, %one : tensor<i32>
  %s268 = stablehlo.add %s267, %one : tensor<i32>
  %s269 = stablehlo.subtract %s268, %one : tensor<i32>
  %s270 = stablehlo.add %s269, %one : tensor<i32>
  %s271 = stablehlo.subtract %s270, %one : tensor<i32>
  %s272 = stablehlo.add %s271, %one : tensor<i32>
  %s273 = stablehlo.subtract %s272, %one : tensor<i32>
  %s274 = stablehlo.add %s273, %one : tensor<i32>
  %s275 = stablehlo.subtract %s274, %one : tensor<i32>
  %s276 = stablehlo.add %s275, %one : tensor<i32>
  %s277 = stablehlo.subtract %s276, %one : tensor<i32>
  %s278 = stablehlo.add %s277, %one : tensor<i32>
  %s279 = stablehlo.subtract %s278, %one : tensor<i32>
  %s280 = stablehlo.add %s279, %one : tensor<i32>
  %s281 = stablehlo.subtract %s280, %one : tensor<i32>
  %s282 = stablehlo.add %s281, %one : tensor<i32>
  %s283 = stablehlo.subtract %s282, %one : tensor<i32>
  %s284 = stablehlo.add %s283, %one : tensor<i32>
  %s285 = stablehlo.subtract %s284, %one : tensor<i32>
  %s286 = stablehlo.add %s285, %one : tensor<i32>
  %s287 = stablehlo.subtract %s286, %one : tensor<i32>
  %s288 = stablehlo.add %s287, %one : tensor<i32>
  %s289 = stablehlo.subtract %s288, %one : tensor<i32>
  %s290 = stablehlo.add %s289, %one : tensor<i32>
  %s291 = stablehlo.subtract %s290, %one : tensor<i32>
  %s292 = stablehlo.add %s291, %one : tensor<i32>
  %s293 = stablehlo.subtract %s292, %one : tensor<i32>
  %s294 = stablehlo.add %s293, %one : tensor<i32>
  %s295 = stablehlo.subtract %s294, %one : tensor<i32>
  %s296 = stablehlo.add %s295, %one : tensor<i32>
  %s297 = stablehlo.subtract %s296, %one : tensor<i32>
  %s298 = stablehlo.add %s297, %one : tensor<i32>
  %s299 = stablehlo.subtract %s298, %one : tensor<i32>
  %s300 = stablehlo.add %s299, %one : tensor<i32>
  %s301 = stablehlo.subtract %s300, %one : tensor<i32>
  %s302 = stablehlo.add %s301, %one : tensor<i32>
  %s303 = stablehlo.subtract %s302, %one : tensor<i32>
  %s304 = stablehlo.add %s303, %one : tensor<i32>
  %s305 = stablehlo.subtract %s304, %one : tensor<i32>
  %s306 = stablehlo.add %s305, %one : tensor<i32>
  %s307 = stablehlo.subtract %s306, %one : tensor<i32>
  %s308 = stablehlo.add %s307, %one : tensor<i32>
  %s309 = stablehlo.subtract %s308, %one : tensor<i32>
  %s310 = stablehlo.add %s309, %one : tensor<i32>
  %s311 = stablehlo.subtract %s310, %one : tensor<i32>
  %s312 = stablehlo.add %s311, %one : tensor<i32>
  %s313 = stablehlo.subtract %s312, %one : tensor<i32>
  %s314 = stablehlo.add %s313, %one : tensor<i32>
  %s315 = stablehlo.subtract %s314, %one : tensor<i32>
  %s316 = stablehlo.add %s315, %one : tensor<i32>
  %s317 = stablehlo.subtract %s316, %one : tensor<i32>
  %s318 = stablehlo.add %s317, %one : tensor<i32>
  %s319 = stablehlo.subtract %s318, %one : tensor<i32>
  %s320 = stablehlo.add %s319, %one : tensor<i32>
  %s321 = stablehlo.subtract %s320, %one : tensor<i32>
  %s322 = stablehlo.add %s321, %one : tensor<i32>
  %s323 = stablehlo.subtract %s322, %one : tensor<i32>
  %s324 = stablehlo.add %s323, %one : tensor<i32>
  %s325 = stablehlo.subtract %s324, %one : tensor<i32>
  %s326 = stablehlo.add %s325, %one : tensor<i32>
  %s327 = stablehlo.subtract %s326, %one : tensor<i32>
  %s328 = stablehlo.add %s327, %one : tensor<i32>
  %s329 = stablehlo.subtract %s328, %one : tensor<i32>
  %s330 = stablehlo.add %s329, %one : tensor<i32>
  %s331 = stablehlo.subtract %s330, %one : tensor<i32>
  %s332 = stablehlo.add %s331, %one : tensor<i32>
  %s333 = stablehlo.subtract %s332, %one : tensor<i32>
  %s334 = stablehlo.add %s333, %one : tensor<i32>
  %s335 = stablehlo.subtract %s334, %one : tensor<i32>
  %s336 = stablehlo.add %s335, %one : tensor<i32>
  %s337 = stablehlo.subtract %s336, %one : tensor<i32>
  %s338 = stablehlo.add %s337, %one : tensor<i32>
  %s339 = stablehlo.subtract %s338, %one : tensor<i32>
  %s340 = stablehlo.add %s339, %one : tensor<i32>
  %s341 = stablehlo.subtract %s340, %one : tensor<i32>
  %s342 = stablehlo.add %s341, %one : tensor<i32>
  %s343 = stablehlo.subtract %s342, %one : tensor<i32>
  %s344 = stablehlo.add %s343, %one : tensor<i32>
  %s345 = stablehlo.subtract %s344, %one : tensor<i32>
  %s346 = stablehlo.add %s345, %one : tensor<i32>
  %s347 = stablehlo.subtract %s346, %one : tensor<i32>
  %s348 = stablehlo.add %s347, %one : tensor<i32>
  %s349 = stablehlo.subtract %s348, %one : tensor<i32>
  %s350 = stablehlo.add %s349, %one : tensor<i32>
  %s351 = stablehlo.subtract %s350, %one : tensor<i32>
  %s352 = stablehlo.add %s351, %one : tensor<i32>
  %s353 = stablehlo.subtract %s352, %one : tensor<i32>
  %s354 = stablehlo.add %s353, %one : tensor<i32>
  %s355 = stablehlo.subtract %s354, %one : tensor<i32>
  %s356 = stablehlo.add %s355, %one : tensor<i32>
  %s357 = stablehlo.subtract %s356, %one : tensor<i32>
  %s358 = stablehlo.add %s357, %one : tensor<i32>
  %s359 = stablehlo.subtract %s358, %one : tensor<i32>
  %s360 = stablehlo.add %s359, %one : tensor<i32>
  %s361 = stablehlo.subtract %s360, %one : tensor<i32>
  %s362 = stablehlo.add %s361, %one : tensor<i32>
  %s363 = stablehlo.subtract %s362, %one : tensor<i32>
  %s364 = stablehlo.add %s363, %one : tensor<i32>
  %s365 = stablehlo.subtract %s364, %one : tensor<i32>
  %s366 = stablehlo.add %s365, %one : tensor<i32>
  %s367 = stablehlo.subtract %s366, %one : tensor<i32>
  %s368 = stablehlo.add %s367, %one : tensor<i32>
  %s369 = stablehlo.subtract %s368, %one : tensor<i32>
  %s370 = stablehlo.add %s369, %one : tensor<i32>
  %s371 = stablehlo.subtract %s370, %one : tensor<i32>
  %s372 = stablehlo.add %s371, %one : tensor<i32>
  %s373 = stablehlo.subtract %s372, %one : tensor<i32>
  %s374 = stablehlo.add %s373, %one : tensor<i32>
  %s375 = stablehlo.subtract %s374, %one : tensor<i32>
  %s376 = stablehlo.add %s375, %one : tensor<i32>
  %s377 = stablehlo.subtract %s376, %one : tensor<i32>
  %s378 = stablehlo.add %s377, %one : tensor<i32>
  %s379 = stablehlo.subtract %s378, %one : tensor<i32>
  %s380 = stablehlo.add %s379, %one : tensor<i32>
  %s381 = stablehlo.subtract %s380, %one : tensor<i32>
  %s382 = stablehlo.add %s381, %one : tensor<i32>
  %s383 = stablehlo.subtract %s382, %one : tensor<i32>
  %s384 = stablehlo.add %s383, %one : tensor<i32>
  %s385 = stablehlo.subtract %s384, %one : tensor<i32>
  %s386 = stablehlo.add %s385, %one : tensor<i32>
  %s387 = stablehlo.subtract %s386, %one : tensor<i32>
  %s388 = stablehlo.add %s387, %one : tensor<i32>
  %s389 = stablehlo.subtract %s388, %one : tensor<i32>
  %s390 = stablehlo.add %s389, %one : tensor<i32>
  %s391 = stablehlo.subtract %s390, %one : tensor<i32>
  %s392 = stablehlo.add %s391, %one : tensor<i32>
  %s393 = stablehlo.subtract %s392, %one : tensor<i32>
  %s394 = stablehlo.add %s393, %one : tensor<i32>
  %s395 = stablehlo.subtract %s394, %one : tensor<i32>
  %s396 = stablehlo.add %s395, %one : tensor<i32>
  %s397 = stablehlo.subtract %s396, %one : tensor<i32>
  %s398 = stablehlo.add %s397, %one : tensor<i32>
  %s399 = stablehlo.subtract %s398, %one : tensor<i32>
  %bound = stablehlo.constant dense<3> : tensor<i32>
  %ok = stablehlo.compare GE, %s399, %bound, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok) {error_message = "b >= 3", has_side_effect = true} : (tensor<i1>) -> ()
  %doubled = stablehlo.add %arg0, %arg0 : tensor<?x4xf32>
  return %doubled : tensor<?x4xf32>
}

// JAX's derived dimension 2*s, checked through private floor-division
// helpers: the even size takes the asserted s range scaled by 2, and the other
// argument, asserted equal to 2*s, takes the same dimension.
// INPUT-LABEL: util.func public @derived_dimension_facts
// INPUT: %[[BOUNDED:.+]] = util.assume.int %{{.+}}<umin = 2, umax = 8, udiv = 2> : i32
// INPUT: %[[DIM:.+]] = arith.index_cast %[[BOUNDED]] : i32 to index
// INPUT-DAG: flow.tensor.tie_shape %{{.+}} : tensor<3x?x5xf32>{%[[DIM]]}
// INPUT-DAG: flow.tensor.tie_shape %{{.+}} : tensor<3x?x5xf32>{%[[DIM]]}
func.func @derived_dimension_facts(%arg0: tensor<3x?x5xf32>, %arg1: tensor<3x?x5xf32>) -> tensor<3x?x5xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %two = stablehlo.constant dense<2> : tensor<i32>
  %four = stablehlo.constant dense<4> : tensor<i32>
  %minusOne = stablehlo.constant dense<-1> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?x5xf32>) -> tensor<i32>
  %otherSize = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<3x?x5xf32>) -> tensor<i32>
  %qr:2 = call @floor_divmod(%size, %two) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %even = stablehlo.compare EQ, %qr#1, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%even, %qr#1) {error_message = "remainder {0}", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %positive = stablehlo.compare GE, %qr#0, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %qr#0) {error_message = "s < 1", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %doubled = stablehlo.multiply %qr#0, %two : tensor<i32>
  %same = stablehlo.compare EQ, %otherSize, %doubled, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same, %otherSize) {error_message = "arg1 != 2*s", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %negated = stablehlo.multiply %qr#0, %minusOne : tensor<i32>
  %belowMax = stablehlo.add %negated, %four : tensor<i32>
  %maxOk = stablehlo.compare GE, %belowMax, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%maxOk, %belowMax) {error_message = "s > 4", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %sum = stablehlo.add %arg0, %arg1 : tensor<3x?x5xf32>
  return %sum : tensor<3x?x5xf32>
}

// Without the zero-remainder check, 2*s may be one less than the size: no tie
// and no range.
// INPUT-LABEL: util.func public @quotient_without_divisibility
// INPUT-NOT: util.assume.int
// INPUT-NOT: flow.tensor.tie_shape
// INPUT: util.return
func.func @quotient_without_divisibility(%arg0: tensor<3x?x5xf32>, %arg1: tensor<3x?x5xf32>) -> tensor<3x?x5xf32> {
  %one = stablehlo.constant dense<1> : tensor<i32>
  %two = stablehlo.constant dense<2> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?x5xf32>) -> tensor<i32>
  %otherSize = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<3x?x5xf32>) -> tensor<i32>
  %qr:2 = call @floor_divmod(%size, %two) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %positive = stablehlo.compare GE, %qr#0, %one, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%positive, %qr#0) {error_message = "s < 1", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %doubled = stablehlo.multiply %qr#0, %two : tensor<i32>
  %same = stablehlo.compare EQ, %otherSize, %doubled, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%same, %otherSize) {error_message = "arg1 != 2*s", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %sum = stablehlo.add %arg0, %arg1 : tensor<3x?x5xf32>
  return %sum : tensor<3x?x5xf32>
}

// The select picks 5 for a size of 0, which passes the check: the predicate
// differs between 0 and positive sizes, so no bound of 3 is recorded.
// INPUT-LABEL: util.func public @undecided_select
// INPUT-NOT: util.assume.int
// INPUT: util.return
func.func @undecided_select(%arg0: tensor<3x?x5xf32>) -> tensor<3x?x5xf32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %three = stablehlo.constant dense<3> : tensor<i32>
  %five = stablehlo.constant dense<5> : tensor<i32>
  %size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<3x?x5xf32>) -> tensor<i32>
  %empty = stablehlo.compare EQ, %size, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %chosen = stablehlo.select %empty, %five, %size : tensor<i1>, tensor<i32>
  %ok = stablehlo.compare GE, %chosen, %three, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%ok, %chosen) {error_message = "{0} < 3", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  %doubled = stablehlo.add %arg0, %arg0 : tensor<3x?x5xf32>
  return %doubled : tensor<3x?x5xf32>
}

func.func private @floor_divmod(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %q = call @floor_divide(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %r = call @floor_remainder(%x, %y) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %q, %r : tensor<i32>, tensor<i32>
}
func.func private @floor_divide(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
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
  return %floor : tensor<i32>
}
func.func private @floor_remainder(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %remainder = stablehlo.remainder %x, %y : tensor<i32>
  %inexact = stablehlo.compare NE, %remainder, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %remainderNegative = stablehlo.compare LT, %remainder, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %yNegative = stablehlo.compare LT, %y, %zero, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %wrongSign = stablehlo.compare NE, %remainderNegative, %yNegative, UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %adjust = stablehlo.and %wrongSign, %inexact : tensor<i1>
  %shifted = stablehlo.add %remainder, %y : tensor<i32>
  %modulo = stablehlo.select %adjust, %shifted, %remainder : tensor<i1>, tensor<i32>
  return %modulo : tensor<i32>
}
