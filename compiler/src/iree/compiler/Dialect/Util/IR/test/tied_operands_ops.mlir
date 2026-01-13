// RUN: iree-opt --split-input-file --verify-diagnostics %s

// expected-error @+1 {{tied_operands index out of range}}
"util.func"() <{function_type = () -> i32, sym_name = "test", sym_visibility = "public", tied_operands = [0 : index]}> ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "util.return"(%0) : (i32) -> ()
}) : () -> ()

// expected-error @+1 {{tied_operands index out of range}}
"util.func"() <{function_type = () -> i32, sym_name = "test", sym_visibility = "public", tied_operands = [-2 : index]}> ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "util.return"(%0) : (i32) -> ()
}) : () -> ()

"util.func"() <{function_type = (!hal.device) -> (i1, i64), sym_name = "test", sym_visibility = "public", tied_operands = [-1 : index]}> ({
^bb0(%arg0: !hal.device):
  %0:2 = "hal.device.query"(%arg0) <{category = "sys", key = "foo"}> : (!hal.device) -> (i1, i64)
  "util.return"(%0#0, %0#1) : (i1, i64) -> ()
}) : () -> ()

// expected-error @+1 {{tied_operands index out of range}}
"util.func"() <{function_type = (!hal.device) -> (i1, i64), sym_name = "test", sym_visibility = "public", tied_operands = [1 : index]}> ({
^bb0(%arg0: !hal.device):
  %0:2 = "hal.device.query"(%arg0) <{category = "sys", key = "foo"}> : (!hal.device) -> (i1, i64)
  "util.return"(%0#0, %0#1) : (i1, i64) -> ()
}) : () -> ()
