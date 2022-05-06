// RUN: iree-opt --split-input-file --iree-hal-pack-dispatch-operands %s | FileCheck %s

stream.executable private @ex0 {
  stream.executable.export public @device_i1
  builtin.module {
    // CHECK-LABEL: func @device_i1
    // CHECK-SAME: (%arg0: i32, %arg1: !stream.binding)
    func.func @device_i1(%arg0: i1 {stream.values = [true, false]}, %arg1: !stream.binding) {
      // CHECK-NEXT: %[[DEV_I1:.+]] = arith.trunci %arg0 {stream.values = [true, false]} : i32 to i1
      // CHECK-NEXT: util.do_not_optimize(%[[DEV_I1]])
      util.do_not_optimize(%arg0) : i1
      return
    }
  }
}
func.func @host_i1(%arg0: i1) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}
  // CHECK: %[[HOST_I32:.+]] = arith.extui %arg0 : i1 to i32
  %1 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_I32]] : i32)
    stream.cmd.dispatch @ex0::@device_i1[%c1, %c1, %c1](%arg0 : i1) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %1 : !stream.timepoint
}

// -----

stream.executable private @ex1 {
  stream.executable.export public @device_bf16
  builtin.module {
    // CHECK-LABEL: func @device_bf16
    // CHECK-SAME: (%arg0: i32, %arg1: !stream.binding)
    func.func @device_bf16(%arg0: bf16, %arg1: !stream.binding) {
      // CHECK-NEXT: %[[DEV_I16:.+]] = arith.trunci %arg0 : i32 to i16
      // CHECK-NEXT: %[[DEV_BF16:.+]] = arith.bitcast %[[DEV_I16]] : i16 to bf16
      // CHECK-NEXT: util.do_not_optimize(%[[DEV_BF16]])
      util.do_not_optimize(%arg0) : bf16
      return
    }
  }
}
func.func @host_bf16(%arg0: bf16) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}
  // CHECK: %[[HOST_I16:.+]] = arith.bitcast %arg0 : bf16 to i16
  // CHECK: %[[HOST_I32:.+]] = arith.extui %[[HOST_I16]] : i16 to i32
  %1 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_I32]] : i32)
    stream.cmd.dispatch @ex1::@device_bf16[%c1, %c1, %c1](%arg0 : bf16) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %1 : !stream.timepoint
}

// -----

stream.executable private @ex2 {
  // CHECK-LABEL: @device_i64
  stream.executable.export public @device_i64
  builtin.module {
    // CHECK-LABEL: func @device_i64
    // CHECK-SAME: (%arg0: i32, %arg1: i32, %arg2: !stream.binding)
    func.func @device_i64(%arg0: i64 {stream.values = [-1 : i64, 0x0000000200000003 : i64]}, %arg1: !stream.binding) {
      // CHECK-DAG: %[[DEV_LO64:.+]] = arith.extui %arg0 : i32 to i64
      // CHECK-DAG: %[[DEV_HI64:.+]] = arith.extui %arg1 : i32 to i64
      // CHECK-DAG: %[[DEV_HISHL:.+]] = arith.shli %[[DEV_HI64]], %c32
      // CHECK-DAG: %[[DEV_I64:.+]] = arith.ori %[[DEV_LO64]], %[[DEV_HISHL]] {stream.values = [-1, 8589934595]}
      // CHECK-NEXT: util.do_not_optimize(%[[DEV_I64]])
      util.do_not_optimize(%arg0) : i64
      return
    }
  }
}
func.func @host_i64(%arg0: i64) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}
  // CHECK-DAG: %[[HOST_LO32:.+]] = arith.trunci %arg0 : i64 to i32
  // CHECK-DAG: %[[HOST_HISHR:.+]] = arith.shrui %arg0, %c32
  // CHECK-DAG: %[[HOST_HI32:.+]] = arith.trunci %[[HOST_HISHR]] : i64 to i32
  %1 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_LO32]], %[[HOST_HI32]] : i32, i32)
    stream.cmd.dispatch @ex2::@device_i64[%c1, %c1, %c1](%arg0 : i64) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %1 : !stream.timepoint
}

// -----

stream.executable private @ex3 {
  stream.executable.export public @device_index
  builtin.module {
    // CHECK-LABEL: func @device_index
    // CHECK-SAME: (%arg0: i32, %arg1: !stream.binding)
    func.func @device_index(%arg0: index {stream.alignment = 16 : index, stream.values = [0 : index, 1234 : index]}, %arg1: !stream.binding) {
      // CHECK: %[[DEV_INDEX:.+]] = arith.index_cast %arg0 {
      // CHECK-SAME:   stream.alignment = 16 : index
      // CHECK-SAME:   stream.values = [0 : index, 1234 : index]
      // CHECK-SAME: } : i32 to index
      // CHECK: util.do_not_optimize(%[[DEV_INDEX]])
      util.do_not_optimize(%arg0) : index
      return
    }
  }
}
func.func @host_index(%arg0: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}
  // CHECK: %[[HOST_I32:.+]] = arith.index_cast %arg0 : index to i32
  %1 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_I32]] : i32)
    stream.cmd.dispatch @ex3::@device_index[%c1, %c1, %c1](%arg0 : index) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %1 : !stream.timepoint
}
