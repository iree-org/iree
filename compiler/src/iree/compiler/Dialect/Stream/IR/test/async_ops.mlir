// RUN: iree-opt --split-input-file %s --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @asyncAlloca
func.func @asyncAlloca(%arg0: index) -> !stream.resource<transient> {
  // CHECK: = stream.async.alloca : !stream.resource<transient>{%arg0}
  %0 = stream.async.alloca : !stream.resource<transient>{%arg0}
  return %0 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncConstant
func.func @asyncConstant(%arg0: index) -> !stream.resource<transient> {
  // CHECK: = stream.async.constant : !stream.resource<transient>{%arg0} = dense<3> : tensor<8xi32>
  %0 = stream.async.constant : !stream.resource<transient>{%arg0} = dense<3> : tensor<8xi32>
  return %0 : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @asyncSplat
func.func @asyncSplat(%arg0: index, %arg1: i32) -> !stream.resource<*> {
  // CHECK: = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
  %0 = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncClone
func.func @asyncClone(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  // CHECK: = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
  %0 = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncSlice
func.func @asyncSlice(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.slice %arg0[%c0 to %c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  %0 = stream.async.slice %arg0[%c0 to %c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c128}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncFill
func.func @asyncFill(%arg0: !stream.resource<*>, %arg1: index, %arg2: i32) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.fill %arg2, %arg0[%c0 to %c128 for %c128] : i32 -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.fill %arg2, %arg0[%c0 to %c128 for %c128] : i32 -> %arg0 as !stream.resource<*>{%arg1}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncUpdate
func.func @asyncUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.update %arg2, %arg0[%c0 to %c128] : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.update %arg2, %arg0[%c0 to %c128] : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncCopy
func.func @asyncCopy(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.copy %arg2[%c0 to %c128], %arg0[%c0 to %c128], %c128 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.async.copy %arg2[%c0 to %c128], %arg0[%c0 to %c128], %c128 : !stream.resource<*>{%arg3} -> %arg0 as !stream.resource<*>{%arg1}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncTransfer
func.func @asyncTransfer(%arg0: !stream.resource<constant>, %arg1: index) -> !stream.resource<staging> {
  // CHECK: = stream.async.transfer %arg0 : !stream.resource<constant>{%arg1} -> !stream.resource<staging>{%arg1}
  %0 = stream.async.transfer %arg0 : !stream.resource<constant>{%arg1} -> !stream.resource<staging>{%arg1}
  return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @asyncLoad
func.func @asyncLoad(%arg0: !stream.resource<staging>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.async.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> f32
  %0 = stream.async.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @asyncStore
func.func @asyncStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.async.store %arg2, %arg0[%c0] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  %0 = stream.async.store %arg2, %arg0[%c0] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @asyncDispatch
func.func @asyncDispatch(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  // CHECK: = stream.async.dispatch @executable::@dispatch[%c1, %c2, %c3](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  %0 = stream.async.dispatch @executable::@dispatch[%c1, %c2, %c3](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  return %0 : !stream.resource<*>
}

// -----

stream.executable private @executable {
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    stream.return %arg0, %arg1, %arg0 : index, index, index
  }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

// CHECK-LABEL: @asyncDispatchWithWorkgroupCount
func.func @asyncDispatchWithWorkgroupCount(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: = stream.async.dispatch @executable::@dispatch[%c1, %c2](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  %0 = stream.async.dispatch @executable::@dispatch[%c1, %c2](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  return %0 : !stream.resource<*>
}

// -----

stream.executable private @executable {
  stream.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    stream.return %arg0, %arg0, %arg0 : index, index, index
  }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

func.func @asyncDispatchWithInvalidWorkload(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{op workload mismatch; entry point expects 1 arguments but dispatch provides 2}}
  %0 = stream.async.dispatch @executable::@dispatch[%c1, %c2](%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncDispatchNoWorkload
func.func @asyncDispatchNoWorkload(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // CHECK: = stream.async.dispatch @executable::@dispatch(%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  %0 = stream.async.dispatch @executable::@dispatch(%arg0[%c0 to %arg1 for %arg1], %c4) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncExecute
func.func @asyncExecute(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK: = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> %arg0{%arg1} {
  %0:2 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) -> %arg0 as !stream.resource<*>{%arg1} {
    // CHECK: %[[W:.+]] = stream.async.concurrent with(%arg3 as %arg4: !stream.resource<*>{%arg1}) -> %arg3{%arg1} {
    %1 = stream.async.concurrent with(%arg3 as %arg4: !stream.resource<*>{%arg1}) -> %arg3 as !stream.resource<*>{%arg1} {
      // CHECK: stream.yield %arg4 : !stream.resource<*>{%arg1}
      stream.yield %arg4 : !stream.resource<*>{%arg1}
    }
    // CHECK: stream.yield %[[W]] : !stream.resource<*>{%arg1}
    stream.yield %1 : !stream.resource<*>{%arg1}
  } => !stream.timepoint
  return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @asyncExecuteNoCaptures
func.func @asyncExecuteNoCaptures(%arg0: index, %arg1: i32) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK: = stream.async.execute with() -> !stream.resource<*>{%arg0} {
  %0:2 = stream.async.execute with() -> !stream.resource<*>{%arg0} {
    // CHECK: %[[T:.+]] = stream.async.splat
    %1 = stream.async.splat %arg1 : i32 -> !stream.resource<*>{%arg0}
    // CHECK: stream.yield %[[T]] : !stream.resource<*>{%arg0}
    stream.yield %1 : !stream.resource<*>{%arg0}
  } => !stream.timepoint
  return %0#0, %0#1 : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @asyncExecuteNoResults
func.func @asyncExecuteNoResults(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.timepoint) -> (!stream.timepoint) {
  // CHECK: = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) {
  %0:1 = stream.async.execute await(%arg2) => with(%arg0 as %arg3: !stream.resource<*>{%arg1}) {
    // CHECK: stream.yield
    stream.yield
  } => !stream.timepoint
  return %0#0 : !stream.timepoint
}
