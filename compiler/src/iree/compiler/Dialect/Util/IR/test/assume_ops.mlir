// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @assume.int.single_assumption
util.func public @assume.int.single_assumption(%arg0 : index) -> index  {
  // CHECK: util.assume.int %arg0<umin = 0> : index
  %0 = util.assume.int %arg0<umin=0> : index
  util.return %0 : index
}

// -----
// CHECK-LABEL: @assume.int.multi_assumption
util.func public @assume.int.multi_assumption(%arg0 : index) -> index  {
  // CHECK: util.assume.int %arg0[<umin = 0>, <udiv = 5>] : index
  %0 = util.assume.int %arg0[<umin=0>, <udiv=5>] : index
  util.return %0 : index
}

// -----
// CHECK-LABEL: @assume.int.multi_operand
util.func public @assume.int.multi_operand(%arg0 : index, %arg1 : i64) -> index, i64  {
  // CHECK: util.assume.int
  // CHECK-NEXT: %arg0[<umin = 0>, <udiv = 5>],
  // CHECK-NEXT: %arg1[<umax = 10>, <udiv = 6>]
  // CHECK-NEXT: : index, i64
  %0:2 = util.assume.int %arg0[<umin=0>, <udiv=5>], %arg1[<umax=10>, <udiv=6>] : index, i64
  util.return %0#0, %0#1 : index, i64
}
