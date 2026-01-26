// RUN: iree-opt --split-input-file --verify-diagnostics %s

util.func public @assume.int.multi_operand(%arg0 : index, %arg1 : i64) -> index, i64  {
  // expected-error @+1 {{expected operand #1 to have 3 assumptions but it has 2}}
  %0:2 = util.assume.int %arg0[<umin=0>, <umax=2>, <udiv=16>], %arg1[<umax=10>, <udiv=6>] : index, i64
  util.return %0#0, %0#1 : index, i64
}

// -----

util.func public @assume.int.multi_operand_broadcast(%arg0 : index, %arg1 : i64) -> index, i64  {
  // It is legal to have a mismatched arity if 1.
  %0:2 = util.assume.int %arg0[<umin=0>], %arg1[<umax=10>, <udiv=6>] : index, i64
  util.return %0#0, %0#1 : index, i64
}

// -----

util.func public @assume.int.multi_operand(%arg0 : index, %arg1 : i64) -> index, i64  {
  // expected-error @+1 {{expected 2 assumption rows to match number of operands}}
  %0:2 = "util.assume.int"(%arg0, %arg1) {
    assumptions = []
  } : (index, i64) -> (index, i64)
  util.return %0#0, %0#1 : index, i64
}

// -----

util.func public @assume.int.multi_operand(%arg0 : index, %arg1 : i64) -> index, i64  {
  // expected-error @+1 {{failed to satisfy constraint}}
  %0:2 = "util.assume.int"(%arg0, %arg1) {
    assumptions = [[32], [32]]
  } : (index, i64) -> (index, i64)
  util.return %0#0, %0#1 : index, i64
}

// -----

util.func public @assume.int.multi_operand(%arg0 : index, %arg1 : i64) -> index, i64  {
  // expected-error @+1 {{failed to satisfy constraint}}
  %0:2 = "util.assume.int"(%arg0, %arg1) {
    assumptions = [32, [32]]
  } : (index, i64) -> (index, i64)
  util.return %0#0, %0#1 : index, i64
}

// -----

// COM: This test must use generic syntax, as custom syntax wouldn't allow a wrong tied operand.
// expected-error @+1 {{result #0 tied to invalid operand index -2}}
"util.func"() <{function_type = (tensor<4xi32>) -> i32, sym_name = "tied_operands_invalid_negative", sym_visibility = "public", tied_operands = [-2 : index]}> ({
^bb0(%arg0: tensor<4xi32>):
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "util.return"(%0) : (i32) -> ()
}) : () -> ()

// -----

// COM: This test must use generic syntax, as custom syntax wouldn't allow a wrong tied operand.
// expected-error @+1 {{result #1 tied to invalid operand index 3}}
"util.func"() <{function_type = (i32, tensor<4xi32>, tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>), sym_name = "tied_operands_out_of_bounds1", sym_visibility = "public", tied_operands = [1 : index, 3 : index]}> ({
^bb0(%arg0 : i32, %arg1: tensor<4xi32>, %arg2: tensor<4xf32>):
  "util.return"(%arg1, %arg2) : (tensor<4xi32>, tensor<4xf32>) -> ()
}) : () -> ()

// -----

// COM: This test must use generic syntax, as custom syntax wouldn't allow a wrong tied operand.
// expected-error @+1 {{result #0 tied to invalid operand index 0}}
"util.func"() <{function_type = () -> i32, sym_name = "tied_operands_out_of_bounds2", sym_visibility = "public", tied_operands = [0 : index]}> ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "util.return"(%0) : (i32) -> ()
}) : () -> ()
