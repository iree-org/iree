// RUN: iree-opt --split-input-file --iree-hal-inline-conversion %s | FileCheck %s

// NOTE: memory control ops are currently ignored as we're executing inline and
// assume coherent memory.

// CHECK-LABEL: @cmdMemoryControl
func.func @cmdMemoryControl(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %fence = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
    stream.cmd.flush %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    stream.cmd.invalidate %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    stream.cmd.discard %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  // CHECK: return %c0
  return %fence : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdFill
// CHECK-SAME: (%[[TARGET:.+]]: !util.buffer, %[[TARGET_SIZE:.+]]: index)
func.func @cmdFill(%target: !stream.resource<transient>, %target_size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 128
  %length = arith.constant 128 : index
  // CHECK-DAG: %[[VALUE:.+]] = arith.constant 255
  %value = arith.constant 255 : i32
  %fence = stream.cmd.execute with(%target as %target_inner: !stream.resource<transient>{%target_size}) {
    // CHECK: util.buffer.fill %[[VALUE]], %[[TARGET]][%c0 for %[[LENGTH]]] : i32 -> !util.buffer{%[[TARGET_SIZE]]}
    stream.cmd.fill %value, %target_inner[%c0 for %length] : i32 -> !stream.resource<transient>{%target_size}
  } => !stream.timepoint
  // CHECK: return %c0
  return %fence : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdCopy
// CHECK-SAME: (%[[SRC:.+]]: !util.buffer, %[[SRC_SIZE:.+]]: index, %[[DST:.+]]: !util.buffer, %[[DST_SIZE:.+]]: index)
func.func @cmdCopy(%src: !stream.resource<transient>, %src_size: index,
                   %dst: !stream.resource<staging>, %dst_size: index) -> !stream.timepoint {
  // CHECK-DAG: %[[SRC_OFFSET:.+]] = arith.constant 100
  %src_offset = arith.constant 100 : index
  // CHECK-DAG: %[[DST_OFFSET:.+]] = arith.constant 200
  %dst_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 128
  %length = arith.constant 128 : index
  %fence = stream.cmd.execute with(%src as %src_inner: !stream.resource<transient>{%src_size},
                                   %dst as %dst_inner: !stream.resource<staging>{%dst_size}) {
    // CHECK: util.buffer.copy %[[SRC]][%[[SRC_OFFSET]]], %[[DST]][%[[DST_OFFSET]]], %[[LENGTH]] : !util.buffer{%[[SRC_SIZE]]} -> !util.buffer{%[[DST_SIZE]]}
    stream.cmd.copy %src_inner[%src_offset], %dst_inner[%dst_offset], %length : !stream.resource<transient>{%src_size} -> !stream.resource<staging>{%dst_size}
  } => !stream.timepoint
  // CHECK: return %c0
  return %fence : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdExecute
func.func @cmdExecute(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %fence = stream.cmd.execute await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
    stream.cmd.concurrent {
      // CHECK: util.buffer.copy
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      // CHECK: util.buffer.copy
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      stream.cmd.serial {
        // CHECK: util.buffer.copy
        stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
        // CHECK: util.buffer.copy
        stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      }
      // CHECK: util.buffer.copy
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
    }
  } => !stream.timepoint
  // CHECK: return %c0
  return %fence : !stream.timepoint
}

// -----

// Provided by the iree-hal-inline-executables pass:
func.func private @__dispatch_ex_dispatch(
    index, index,                 // workload[2]
    i32, i32,                     // push_constants[2]
    !util.buffer, !util.buffer,   // bindingBuffers[2]
    index, index,                 // bindingOffsets[2]
    index, index)                 // bindingLengths[2]

// NOTE: %buffer0 is transient and will map to a !util.buffer, while
//       %buffer1 is external and will map to a !hal.buffer.

// CHECK-LABEL: @cmdDispatch
// CHECK-SAME: (%[[BUFFER0:.+]]: !util.buffer, %[[BUFFER0_SIZE:.+]]: index,
// CHECK-SAME:  %[[BUFFER1:.+]]: !hal.buffer, %[[BUFFER1_SIZE:.+]]: index)
func.func @cmdDispatch(%buffer0: !stream.resource<transient>, %buffer0_size: index,
                       %buffer1: !stream.resource<external>, %buffer1_size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c128 = arith.constant 128 : index
  // CHECK: %[[BUFFER0_REL_OFFSET:.+]] = arith.constant 200
  %buffer0_offset = arith.constant 200 : index
  // CHECK: %[[BUFFER1_REL_OFFSET:.+]] = arith.constant 300
  %buffer1_offset = arith.constant 300 : index
  %fence = stream.cmd.execute with(%buffer0 as %buffer0_inner: !stream.resource<transient>{%buffer0_size},
                                   %buffer1 as %buffer1_inner: !stream.resource<external>{%buffer1_size}) {
    // CHECK: %[[BUFFER1_STORAGE:.+]] = hal_inline.buffer.storage<%[[BUFFER1]]
    // CHECK: call @__dispatch_ex_dispatch(
    // CHECK-SAME: %c1, %c2,
    // CHECK-SAME: %c4_i32, %c5_i32,
    // CHECK-SAME: %[[BUFFER0]], %[[BUFFER1_STORAGE]],
    // CHECK-SAME: %[[BUFFER0_REL_OFFSET]], %[[BUFFER1_REL_OFFSET]],
    // CHECK-SAME: %c128, %c128)
    stream.cmd.dispatch @ex::@dispatch[%c1, %c2](%c4_i32, %c5_i32 : i32, i32) {
      ro %buffer0_inner[%buffer0_offset for %c128] : !stream.resource<transient>{%buffer0_size},
      wo %buffer1_inner[%buffer1_offset for %c128] : !stream.resource<external>{%buffer1_size}
    } attributes {
      // From the iree-hal-inline-executables pass:
      hal_inline.target = @__dispatch_ex_dispatch
    }
  } => !stream.timepoint
  // CHECK: return %c0
  return %fence : !stream.timepoint
}
