// RUN: iree-opt --split-input-file --iree-stream-annotate-dispatch-assumptions %s | FileCheck %s

// Tests that operands are annotated with their potential values.
// %arg0: ranges known (0) for first call site, no annotations for second, range.div known for third.
// %arg1: ranges and divisibility known at all three call sites.
// %arg2: ranges known at all three call sites. No divisibility because 0/1.
// %arg3: don't yet support floats.
// %arg4: No analysis information known so should not be included in assume.

// CHECK-LABEL: @analyzeAssumptionsEx
stream.executable private @analyzeAssumptionsEx {
  stream.executable.export public @dispatch
  builtin.module  {
    // CHECK: util.func public @dispatch(
    // CHECK: %[[ASSUME:.*]]:3 = util.assume.int
    // CHECK-NEXT:   %arg0[<umin = 0, umax = 0>, <>, <umin = 5, umax = 5, udiv = 5>]
    // CHECK-NEXT:   %arg1[<umin = 20, umax = 20, udiv = 20>, <umin = 40, umax = 40, udiv = 40>, <umin = 20, umax = 20, udiv = 20>]
    // CHECK-NEXT:   %arg2[<umin = 0, umax = 0>, <umin = 1, umax = 1>, <umin = 0, umax = 0>]
    // CHECK-NEXT: : i32, index, i1
    util.func public @dispatch(%arg0: i32, %arg1: index, %arg2: i1, %arg3: f32, %arg4 : i32, %binding: !stream.binding) {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg1, %arg1 : index
      %2 = arith.andi %arg2, %arg2 : i1
      util.return
    }
  }
}
util.func public @analyzeAssumptionsFunc1(%arg0: i32, %arg1: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1
  %c500 = arith.constant 500.0 : f32
  %c600 = arith.constant 600.0 : f32
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c1}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c1}) {
    stream.cmd.dispatch @analyzeAssumptionsEx::@dispatch[%c1, %c1, %c1](%c0_i32, %c20, %false, %c500, %arg1 : i32, index, i1, f32, i32) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
    stream.cmd.dispatch @analyzeAssumptionsEx::@dispatch[%c1, %c1, %c1](%arg0, %c40, %true, %c600, %arg1 : i32, index, i1, f32, i32) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
  } => !stream.timepoint
  util.return
}

util.func public @analyzeAssumptionsFunc2(%arg0: i32, %arg1 : i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5_i32 = arith.constant 5 : i32
  %c128 = arith.constant 20 : index
  %false = arith.constant 0 : i1
  %c500 = arith.constant 500.0 : f32
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c1}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c1}) {
    stream.cmd.dispatch @analyzeAssumptionsEx::@dispatch[%c1, %c1, %c1](%c5_i32, %c128, %false, %c500, %arg1 : i32, index, i1, f32, i32) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
  } => !stream.timepoint
  util.return
}
