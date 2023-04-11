// RUN: iree-opt --split-input-file --allow-unregistered-dialect %s | iree-opt --split-input-file --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @cmdMemoryControl
func.func @cmdMemoryControl(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
    // CHECK: stream.cmd.flush %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    stream.cmd.flush %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    // CHECK: stream.cmd.invalidate %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    stream.cmd.invalidate %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    // CHECK: stream.cmd.discard %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    stream.cmd.discard %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdFill
func.func @cmdFill(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32
  %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
    // CHECK: stream.cmd.fill %c255_i32, %arg2[%c0 for %c128] : i32 -> !stream.resource<transient>{%arg1}
    stream.cmd.fill %c255_i32, %arg2[%c0 for %c128] : i32 -> !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdCopy
func.func @cmdCopy(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %0 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<transient>{%arg1}, %arg2 as %arg5: !stream.resource<staging>{%arg3}) {
    // CHECK: stream.cmd.copy %arg4[%c0], %arg5[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
    stream.cmd.copy %arg4[%c0], %arg5[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
  } => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdCollective
func.func @cmdCollective(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<transient>, %arg3: index) -> !stream.timepoint {
  // CHECK: %[[CHANNEL:.+]] = stream.channel.default
  %channel = stream.channel.default : !stream.channel

  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: stream.cmd.execute
  %0 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<transient>{%arg1}, %arg2 as %arg5: !stream.resource<transient>{%arg3}) {

    // Out-of-place all-reduce:
    // CHECK-NEXT: stream.cmd.collective<all_reduce with sum : f32>[%c128] channel(%[[CHANNEL]]) {
    // CHECK-NEXT:   ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
    // CHECK-NEXT:   wo %arg5[%c0 for %c128] : !stream.resource<transient>{%arg3}
    // CHECK-NEXT: }
    stream.cmd.collective<all_reduce with sum : f32>[%c128] channel(%channel) {
      ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
      wo %arg5[%c0 for %c128] : !stream.resource<transient>{%arg3}
    }

    // In-place all-reduce:
    // CHECK-NEXT: stream.cmd.collective<all_reduce with average : f32>[%c128] channel(%[[CHANNEL]]) {
    // CHECK-NEXT:   ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
    // CHECK-NEXT:   wo %arg4[%c0 for %c128] : !stream.resource<transient>{%arg3}
    // CHECK-NEXT: }
    stream.cmd.collective<all_reduce with average : f32>[%c128] channel(%channel) {
      ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
      wo %arg4[%c0 for %c128] : !stream.resource<transient>{%arg3}
    }

    // Send:
    // CHECK-NEXT: stream.cmd.collective<send : f32>[%c128] channel(%[[CHANNEL]]) {
    // CHECK-NEXT:   ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1}
    // CHECK-NEXT: }
    stream.cmd.collective<send : f32>[%c128] channel(%channel) {
      ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1}
    }

    // Recv:
    // CHECK-NEXT: stream.cmd.collective<recv : f32>[%c128] channel(%[[CHANNEL]]) {
    // CHECK-NEXT:   wo %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1}
    // CHECK-NEXT: }
    stream.cmd.collective<recv : f32>[%c128] channel(%channel) {
      wo %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1}
    }

  } => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdDispatch
func.func @cmdDispatch(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<external>, %arg3: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c128 = arith.constant 128 : index
  %0 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<transient>{%arg1}, %arg2 as %arg5: !stream.resource<external>{%arg3}) {
    //      CHECK: stream.cmd.dispatch {@executable::@dispatch0, @executable::@dispatch1}[%c1, %c2, %c3](%c4, %c5 : index, index) {
    // CHECK-NEXT:   ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
    // CHECK-NEXT:   wo %arg5[%c0 for %c128] : !stream.resource<external>{%arg3}
    // CHECK-NEXT: }
    stream.cmd.dispatch {@executable::@dispatch0, @executable::@dispatch1}[%c1, %c2, %c3](%c4, %c5 : index, index) {
      ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
      wo %arg5[%c0 for %c128] : !stream.resource<external>{%arg3}
    }
  } => !stream.timepoint
  return %0 : !stream.timepoint
}

// -----

// CHECK: stream.cmd.func private @cmdFunc(%arg0[%arg1 for %arg2]: !stream.resource<*>, %arg3: i32, %arg4[%arg5 for %arg6]: !stream.resource<*>, %arg7: !custom.type, %arg8[%arg9 for %arg10]: !stream.resource<*>)
stream.cmd.func private @cmdFunc(%arg0[%arg1 for %arg2]: !stream.resource<*>, %arg3: i32, %arg4[%arg5 for %arg6]: !stream.resource<*>, %arg7: !custom.type, %arg8[%arg9 for %arg10]: !stream.resource<*>)

// CHECK-LABEL: func.func @cmdCall
func.func @cmdCall(%arg0: !stream.resource<external>, %arg1: i32, %arg2: !stream.resource<external>, %arg3: !custom.type, %arg4: !stream.resource<external>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %size0 = arith.constant 100 : index
  %size1 = arith.constant 101 : index
  %size2 = arith.constant 102 : index
  // CHECK: stream.cmd.execute with(%arg0 as %[[STREAM0:.+]]: !stream.resource<external>{%[[SIZE0:.+]]}, %arg2 as %[[STREAM1:.+]]: !stream.resource<external>{%[[SIZE1:.+]]}, %arg4 as %[[STREAM2:.+]]: !stream.resource<external>{%[[SIZE2:.+]]}) {
  %timepoint = stream.cmd.execute with(%arg0 as %stream0: !stream.resource<external>{%size0}, %arg2 as %stream1: !stream.resource<external>{%size1}, %arg4 as %stream2: !stream.resource<external>{%size2}) {
    // CHECK: stream.cmd.call @cmdFunc(ro %[[STREAM0]][%c0 for %[[SIZE0]]], %arg1, rw %[[STREAM1]][%c0 for %[[SIZE1]]], %arg3, wo %[[STREAM2]][%c0 for %[[SIZE2]]]) : (!stream.resource<external>{%[[SIZE0]]}, i32, !stream.resource<external>{%[[SIZE1]]}, !custom.type, !stream.resource<external>{%[[SIZE2]]}) -> ()
    stream.cmd.call @cmdFunc(ro %stream0[%c0 for %size0], %arg1, rw %stream1[%c0 for %size1], %arg3, wo %stream2[%c0 for %size2]) : (!stream.resource<external>{%size0}, i32, !stream.resource<external>{%size1}, !custom.type, !stream.resource<external>{%size2}) -> ()
  } => !stream.timepoint
  return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdExecute
func.func @cmdExecute(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.cmd.execute await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
  %0 = stream.cmd.execute await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
    // CHECK: stream.cmd.concurrent {
    stream.cmd.concurrent {
      // CHECK-NEXT: stream.cmd.copy
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      // CHECK-NEXT: stream.cmd.copy
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      // CHECK: stream.cmd.serial {
      stream.cmd.serial {
        // CHECK-NEXT: stream.cmd.copy
        stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
        // CHECK-NEXT: stream.cmd.copy
        stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      }
    }
  // CHECK: } => !stream.timepoint
  } => !stream.timepoint
  return %0 : !stream.timepoint
}
