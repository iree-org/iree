// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldSubviewsIntoCmdTOp
util.func private @FoldSubviewsIntoCmdTOp(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1000 = arith.constant 1000 : index
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  %c255_i32 = arith.constant 255 : i32
  %0 = stream.resource.subview %arg0[%c64] : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%c3000}
  %1 = stream.cmd.execute with(%0 as %arg2: !stream.resource<transient>{%arg1}) {
    // CHECK: stream.cmd.flush %arg2[%c1064 for %c2000] : !stream.resource<transient>{%arg1}
    stream.cmd.flush %arg2[%c1000 for %c2000] : !stream.resource<transient>{%c3000}
    // CHECK: stream.cmd.invalidate %arg2[%c1064 for %c2000] : !stream.resource<transient>{%arg1}
    stream.cmd.invalidate %arg2[%c1000 for %c2000] : !stream.resource<transient>{%c3000}
    // CHECK: stream.cmd.discard %arg2[%c1064 for %c2000] : !stream.resource<transient>{%arg1}
    stream.cmd.discard %arg2[%c1000 for %c2000] : !stream.resource<transient>{%c3000}
    // CHECK: stream.cmd.fill %c255_i32, %arg2[%c1064 for %c2000] : i32 -> !stream.resource<transient>{%arg1}
    stream.cmd.fill %c255_i32, %arg2[%c1000 for %c2000] : i32 -> !stream.resource<transient>{%c3000}
  } => !stream.timepoint
  util.return %1 : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldSubviewsIntoCmdCopyOp
util.func private @FoldSubviewsIntoCmdCopyOp(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c1000 = arith.constant 1000 : index
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  %c4000 = arith.constant 4000 : index
  %0 = stream.resource.subview %arg0[%c64] : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%c3000}
  %1 = stream.resource.subview %arg0[%c128] : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%c4000}
  %2 = stream.cmd.execute with(%0 as %arg2: !stream.resource<transient>{%c3000}, %1 as %arg3: !stream.resource<transient>{%c4000}) {
    // CHECK: stream.cmd.copy %arg2[%c1064], %arg2[%c2128], %c1000 : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%arg1}
    stream.cmd.copy %arg2[%c1000], %arg3[%c2000], %c1000 : !stream.resource<transient>{%c3000} -> !stream.resource<transient>{%c4000}
  } => !stream.timepoint
  util.return %2 : !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldSubviewsIntoCmdDispatchOp
util.func private @FoldSubviewsIntoCmdDispatchOp(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c1000 = arith.constant 1000 : index
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  %c4000 = arith.constant 4000 : index
  %0 = stream.resource.subview %arg0[%c64] : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%c3000}
  %1 = stream.resource.subview %arg0[%c128] : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%c4000}
  %2 = stream.cmd.execute with(%0 as %arg2: !stream.resource<transient>{%c3000}, %1 as %arg3: !stream.resource<transient>{%c4000}) {
    // CHECK: stream.cmd.dispatch
    stream.cmd.dispatch @executable::@dispatch[%c1, %c1, %c1] {
      // CHECK-NEXT: ro %arg2[%c1064 for %c1000] : !stream.resource<transient>{%arg1}
      ro %arg2[%c1000 for %c1000] : !stream.resource<transient>{%c3000},
      // CHECK-NEXT: wo %arg2[%c2128 for %c1000] : !stream.resource<transient>{%arg1}
      wo %arg3[%c2000 for %c1000] : !stream.resource<transient>{%c4000}
    }
  } => !stream.timepoint
  util.return %2 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateCmdExecuteWaits
util.func private @ElideImmediateCmdExecuteWaits(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: stream.cmd.execute with
  %0 = stream.cmd.execute await(%imm) => with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
    stream.cmd.discard %arg2[%c0 for %arg1] : !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ChainCmdExecuteWaits
util.func private @ChainCmdExecuteWaits(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: stream.timepoint.await
  %0 = stream.timepoint.await %arg2 => %arg0 : !stream.resource<transient>{%arg1}
  // CHECK: stream.cmd.execute await(%arg2) => with
  %1 = stream.cmd.execute with(%0 as %arg3: !stream.resource<transient>{%arg1}) {
    // CHECK: stream.cmd.discard
    stream.cmd.discard %arg3[%c0 for %c128] : !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  util.return %1 : !stream.timepoint
}

// -----

// CHECK-LABEL: @CloneCapturedCmdExecuteSubviewOps
util.func private @CloneCapturedCmdExecuteSubviewOps(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c1000 = arith.constant 1000 : index
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  // CHECK-NOT: stream.resource.subview
  %0 = stream.resource.subview %arg0[%c64] : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%c3000}
  // CHECK: = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1})
  %1 = stream.cmd.execute with(%0 as %arg3: !stream.resource<transient>{%c3000}) {
    // CHECK: stream.cmd.discard %arg2[%c1064 for %c2000] : !stream.resource<transient>{%arg1}
    stream.cmd.discard %arg3[%c1000 for %c2000] : !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  util.return %1 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideNoOpCmdExecuteOp
util.func private @ElideNoOpCmdExecuteOp(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.timepoint) -> !stream.timepoint {
  // CHECK-NOT: stream.cmd.execute
  %0 = stream.cmd.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<transient>{%arg1}) {
  } => !stream.timepoint
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  // CHECK: util.return %[[IMM]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideUnusedCmdExecuteOp
util.func private @ElideUnusedCmdExecuteOp(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: stream.cmd.execute
  %0 = stream.cmd.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<transient>{%arg1}) {
    stream.cmd.discard %arg3[%c0 for %c128] : !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  util.return
}
