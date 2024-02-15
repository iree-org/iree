// RUN: iree-opt --split-input-file --iree-stream-annotate-dispatch-arguments %s | FileCheck %s

// Tests that external executables don't get annotated

// CHECK-LABEL: @skipExternExecutablesEx
stream.executable private @skipExternExecutablesEx {
  // CHECK: stream.executable.export public @dispatch
  stream.executable.export public @dispatch
}
util.func public @skipExternExecutables(%arg0: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c1}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c1}) {
    stream.cmd.dispatch @annotatePotentialValuesEx::@dispatch[%c1, %c1, %c1](%c0_i32 : i32) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Tests that operands are annotated with their potential values.
// %arg0: cannot be annotated because it comes from outside the program.
// %arg1: all values known, gets alignment being an index.
// %arg2: all values known, no alignment (doesn't make sense for i1).
// %arg3: don't yet support floats.

// CHECK-LABEL: @annotatePotentialValuesEx
stream.executable private @annotatePotentialValuesEx {
  stream.executable.export public @dispatch
  builtin.module  {
    // CHECK:  util.func public @dispatch(
    // CHECK-SAME: %arg0: i32,
    // CHECK-SAME: %arg1: index {stream.alignment = 4 : index, stream.values = [20 : index, 40 : index]},
    // CHECK-SAME: %arg2: i1 {stream.values = [false, true]},
    // CHECK-SAME: %arg3: f32
     util.func public @dispatch(%arg0: i32, %arg1: index, %arg2: i1, %arg3: f32, %binding: !stream.binding) {
      util.return
    }
  }
}
util.func public @annotatePotentialValues(%arg0: i32) {
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
    stream.cmd.dispatch @annotatePotentialValuesEx::@dispatch[%c1, %c1, %c1](%c0_i32, %c20, %false, %c500 : i32, index, i1, f32) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
    stream.cmd.dispatch @annotatePotentialValuesEx::@dispatch[%c1, %c1, %c1](%arg0, %c40, %true, %c600 : i32, index, i1, f32) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Tests that index operands are analyzed for alignment.
// %arg0: not analyzable as %arg0 comes from outside the program.
// %arg1: all values aren't known but the util.align gives us what we need.
// %arg2: all values are known but unaligned.
// %arg3: all values are known and have a derivable alignment.
// %arg4: global initialized to 1024 and may be set to 2048.

// CHECK-LABEL: @annotateOperandAlignmentEx
stream.executable private @annotateOperandAlignmentEx {
  stream.executable.export public @dispatch
  builtin.module  {
    // CHECK:  util.func public @dispatch(
    // CHECK-SAME: %arg0: index,
    // CHECK-SAME: %arg1: index {stream.alignment = 16 : index},
    // CHECK-SAME: %arg2: index {stream.values = [4096 : index, 4097 : index]},
    // CHECK-SAME: %arg3: index {stream.alignment = 16 : index, stream.values = [1200 : index, 5232 : index]}
    // CHECK-SAME: %arg4: index {stream.alignment = 1024 : index, stream.values = [1024 : index, 2048 : index]}
     util.func public @dispatch(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %binding: !stream.binding) {
      util.return
    }
  }
}
util.global private mutable @global_var = 1024 : index
util.func public @otherFunc() {
  %c2048 = arith.constant 2048 : index
  util.global.store %c2048, @global_var : index
  util.return
}
util.func public @annotateOperandAlignment(%arg0: index, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c4096 = arith.constant 4096 : index
  %c4097 = arith.constant 4097 : index
  %c1200 = arith.constant 1200 : index
  %c5232 = arith.constant 5232 : index
  %aligned1 = util.align %arg1, %c16 : index
  %global_value = util.global.load @global_var : index
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c1}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c1}) {
    stream.cmd.dispatch @annotateOperandAlignmentEx::@dispatch[%c1, %c1, %c1](%arg0, %c32, %c4097, %c1200, %global_value : index, index, index, index, index) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
    stream.cmd.dispatch @annotateOperandAlignmentEx::@dispatch[%c1, %c1, %c1](%c16, %aligned1, %c4096, %c5232, %global_value: index, index, index, index, index) {
      rw %capture[%c0 for %c1] : !stream.resource<transient>{%c1}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Tests that resource offset alignment gets tagged on binding arguments.
// %arg0: all values known (including 0), max alignment.
// %arg1: value comes from outside the program so no alignment.
// %arg2: all values known.
// %arg3: util.align provides info required for external value.

// CHECK-LABEL: @annotateBindingAlignmentEx
stream.executable private @annotateBindingAlignmentEx {
  stream.executable.export public @dispatch
  builtin.module  {
    // CHECK:  util.func public @dispatch(
    // CHECK-SAME: %arg0: !stream.binding {stream.alignment = 64 : index},
    // CHECK-SAME: %arg1: !stream.binding,
    // CHECK-SAME: %arg2: !stream.binding {stream.alignment = 8 : index},
    // CHECK-SAME: %arg3: !stream.binding {stream.alignment = 16 : index})
     util.func public @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) {
      util.return
    }
  }
}
util.func public @annotateBindingAlignment(%arg0: index, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c48 = arith.constant 48 : index
  %c64 = arith.constant 64 : index
  %aligned1 = util.align %arg1, %c16 : index
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c64}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c64}) {
    stream.cmd.dispatch @annotateBindingAlignmentEx::@dispatch[%c1, %c1, %c1] {
      rw %capture[%c0 for %c8] : !stream.resource<transient>{%c64},
      rw %capture[%arg0 for %c8] : !stream.resource<transient>{%c64},
      rw %capture[%c8 for %c8] : !stream.resource<transient>{%c64},
      rw %capture[%c32 for %c8] : !stream.resource<transient>{%c64}
    }
    stream.cmd.dispatch @annotateBindingAlignmentEx::@dispatch[%c1, %c1, %c1] {
      rw %capture[%c64 for %c8] : !stream.resource<transient>{%c64},
      rw %capture[%c1 for %c8] : !stream.resource<transient>{%c64},
      rw %capture[%c16 for %c8] : !stream.resource<transient>{%c64},
      rw %capture[%aligned1 for %c8] : !stream.resource<transient>{%c64}
    }
  } => !stream.timepoint
  util.return
}
