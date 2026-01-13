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

// expected-error @+1 {{tied_operands index out of range}}
"util.func"() <{function_type = () -> i32, sym_name = "test", sym_visibility = "public", tied_operands = [0 : index]}> ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "util.return"(%0) : (i32) -> ()
}) : () -> ()

// -----

// expected-error @+1 {{tied_operands index out of range}}
"util.func"() <{function_type = () -> i32, sym_name = "test", sym_visibility = "public", tied_operands = [-2 : index]}> ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "util.return"(%0) : (i32) -> ()
}) : () -> ()

// -----

"util.func"() <{function_type = (!hal.device) -> (i1, i64), sym_name = "test", sym_visibility = "public", tied_operands = [-1 : index]}> ({
^bb0(%arg0: !hal.device):
  %0:2 = "hal.device.query"(%arg0) <{category = "sys", key = "foo"}> : (!hal.device) -> (i1, i64)
  "util.return"(%0#0, %0#1) : (i1, i64) -> ()
}) : () -> ()

// -----

// expected-error @+1 {{tied_operands index out of range}}
"util.func"() <{function_type = (!hal.device) -> (i1, i64), sym_name = "test", sym_visibility = "public", tied_operands = [1 : index]}> ({
^bb0(%arg0: !hal.device):
  %0:2 = "hal.device.query"(%arg0) <{category = "sys", key = "foo"}> : (!hal.device) -> (i1, i64)
  "util.return"(%0#0, %0#1) : (i1, i64) -> ()
}) : () -> ()
