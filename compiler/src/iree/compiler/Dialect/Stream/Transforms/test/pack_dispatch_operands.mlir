// RUN: iree-opt --split-input-file --iree-stream-pack-dispatch-operands %s | FileCheck %s

stream.executable private @ex0 {
  stream.executable.export public @device_i1
  builtin.module {
    // CHECK-LABEL: func.func @device_i1
    // CHECK-SAME: (%arg0: i32, %arg1: !stream.binding)
    func.func @device_i1(%arg0: i1 {stream.values = [true, false]}, %arg1: !stream.binding) {
      // CHECK-NEXT: %[[DEV_I1:.+]] = arith.trunci %arg0 {stream.values = [true, false]} : i32 to i1
      // CHECK-NEXT: util.optimization_barrier %[[DEV_I1]]
      util.optimization_barrier %arg0 : i1
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
    // CHECK-LABEL: func.func @device_bf16
    // CHECK-SAME: (%arg0: i32, %arg1: !stream.binding)
    func.func @device_bf16(%arg0: bf16, %arg1: !stream.binding) {
      // CHECK-NEXT: %[[DEV_I16:.+]] = arith.trunci %arg0 : i32 to i16
      // CHECK-NEXT: %[[DEV_BF16:.+]] = arith.bitcast %[[DEV_I16]] : i16 to bf16
      // CHECK-NEXT: util.optimization_barrier %[[DEV_BF16]]
      util.optimization_barrier %arg0 : bf16
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
    // CHECK-LABEL: func.func @device_i64
    // CHECK-SAME: (%[[DEV_LO32:.+]]: i32, %[[DEV_HI32:.+]]: i32, %arg2: !stream.binding)
    func.func @device_i64(%arg0: i64 {stream.values = [-1 : i64, 0x0000000200000003 : i64]}, %arg1: !stream.binding) {
      // CHECK-DAG: %[[DEV_LO64:.+]] = arith.extui %[[DEV_LO32]] : i32 to i64
      // CHECK-DAG: %[[DEV_HI64:.+]] = arith.extui %[[DEV_HI32]] : i32 to i64
      // CHECK-DAG: %[[DEV_HISHL:.+]] = arith.shli %[[DEV_HI64]], %c32
      // CHECK-DAG: %[[DEV_I64:.+]] = arith.ori %[[DEV_LO64]], %[[DEV_HISHL]] {stream.values = [-1, 8589934595]}
      // CHECK-NEXT: util.optimization_barrier %[[DEV_I64]]
      util.optimization_barrier %arg0 : i64
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

#resourceIndex32 = #stream.resource_config<{
  max_allocation_size = 16,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 32
}>

stream.executable private @ex3 attributes {stream.resources = #resourceIndex32} {
  stream.executable.export public @device_index_32
  builtin.module {
    // CHECK-LABEL: func.func @device_index_32
    // CHECK-SAME: (%[[DEV_I32:.+]]: i32, %{{.+}}: !stream.binding)
    func.func @device_index_32(%arg0: index {stream.alignment = 16 : index, stream.values = [0 : index, 1234 : index]}, %arg1: !stream.binding) {
      // 32-bit device size fits in a push constant:
      // CHECK: %[[DEV_INDEX:.+]] = arith.index_castui %[[DEV_I32]] {
      // CHECK-SAME:   stream.alignment = 16 : index
      // CHECK-SAME:   stream.values = [0 : index, 1234 : index]
      // CHECK-SAME: } : i32 to index
      // CHECK: util.optimization_barrier %[[DEV_INDEX]]
      util.optimization_barrier %arg0 : index
      return
    }
  }
}
func.func @host_index_32(%arg0: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}

  // 32-bit device size fits in a push constant:
  // CHECK: %[[HOST_I32:.+]] = arith.index_castui %arg0 : index to i32
  // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_I32]] : i32)

  %1 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    stream.cmd.dispatch @ex3::@device_index_32[%c1, %c1, %c1](%arg0 : index) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %1 : !stream.timepoint
}

// -----

#resourceIndex64 = #stream.resource_config<{
  max_allocation_size = 16,
  min_buffer_offset_alignment = 16,
  max_buffer_range = 1073741824,
  min_buffer_range_alignment = 16,
  index_bits = 64
}>

stream.executable private @ex4 attributes {stream.resources = #resourceIndex64} {
  stream.executable.export public @device_index_64
  builtin.module {
    // CHECK-LABEL: func.func @device_index_64
    // CHECK-SAME: (%[[DEV_LO32:.+]]: i32, %[[DEV_HI32:.+]]: i32, %{{.+}}: !stream.binding)
    func.func @device_index_64(%arg0: index {stream.alignment = 16 : index, stream.values = [0 : index, 1234 : index]}, %arg1: !stream.binding) {
      // 64-bit device size requires joining after it was split into lo/hi:
      // CHECK-DAG: %[[DEV_LO64:.+]] = arith.extui %[[DEV_LO32]] : i32 to i64
      // CHECK-DAG: %[[DEV_HI64:.+]] = arith.extui %[[DEV_HI32]] : i32 to i64
      // CHECK-DAG: %[[DEV_HISHL:.+]] = arith.shli %[[DEV_HI64]], %c32
      // CHECK-DAG: %[[DEV_I64:.+]] = arith.ori %[[DEV_LO64]], %[[DEV_HISHL]] : i64
      // CHECK-DAG: %[[DEV_INDEX:.+]] = arith.index_castui %[[DEV_I64]] {
      // CHECK-SAME:   stream.alignment = 16 : index
      // CHECK-SAME:   stream.values = [0 : index, 1234 : index]
      // CHECK-SAME: } : i64 to index
      // CHECK: util.optimization_barrier %[[DEV_INDEX]]
      util.optimization_barrier %arg0 : index
      return
    }
  }
}
func.func @host_index_64(%arg0: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}

  // 64-bit device size requires splitting into lo/hi:
  // CHECK: %[[HOST_I64:.+]] = arith.index_castui %arg0 : index to i64
  // CHECK: %[[HOST_LO32:.+]] = arith.trunci %[[HOST_I64]] : i64 to i32
  // CHECK: %[[HOST_HI64:.+]] = arith.shrui %[[HOST_I64]], %c32
  // CHECK: %[[HOST_HI32:.+]] = arith.trunci %[[HOST_HI64]] : i64 to i32
  // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_LO32]], %[[HOST_HI32]] : i32, i32)

  %1 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    stream.cmd.dispatch @ex4::@device_index_64[%c1, %c1, %c1](%arg0 : index) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %1 : !stream.timepoint
}

// -----

stream.executable private @ex5 {
  // CHECK-LABEL: @device_complex_f32
  stream.executable.export public @device_complex_f32
  builtin.module {
    // CHECK-LABEL: func.func @device_complex_f32
    // CHECK-SAME: (%[[DEV_REAL_I32:.+]]: i32, %[[DEV_IMAG_I32:.+]]: i32, %arg2: !stream.binding)
    func.func @device_complex_f32(%arg0: complex<f32>, %arg1: !stream.binding) {
      // CHECK-DAG: %[[DEV_REAL_F32:.+]] = arith.bitcast %[[DEV_REAL_I32]] : i32 to f32
      // CHECK-DAG: %[[DEV_IMAG_F32:.+]] = arith.bitcast %[[DEV_IMAG_I32]] : i32 to f32
      // CHECK-DAG: %[[DEV_COMPLEX:.+]] = complex.create %[[DEV_REAL_F32]], %[[DEV_IMAG_F32]]
      // CHECK-NEXT: util.optimization_barrier %[[DEV_COMPLEX]]
      util.optimization_barrier %arg0 : complex<f32>
      return
    }
  }
}
func.func @host_complex_f32(%arg0: complex<f32>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}
  // CHECK-DAG: %[[HOST_REAL_F32:.+]] = complex.re %arg0
  // CHECK-DAG: %[[HOST_IMAG_F32:.+]] = complex.im %arg0
  // CHECK-DAG: %[[HOST_REAL_I32:.+]] = arith.bitcast %[[HOST_REAL_F32]] : f32 to i32
  // CHECK-DAG: %[[HOST_IMAG_I32:.+]] = arith.bitcast %[[HOST_IMAG_F32]] : f32 to i32
  %1 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_REAL_I32]], %[[HOST_IMAG_I32]] : i32, i32)
    stream.cmd.dispatch @ex5::@device_complex_f32[%c1, %c1, %c1](%arg0 : complex<f32>) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %1 : !stream.timepoint
}

// -----

stream.executable private @ex5 {
  // CHECK-LABEL: @device_complex_f32_bitcast
  stream.executable.export public @device_complex_f32_bitcast
  builtin.module {
    // CHECK-LABEL: func.func @device_complex_f32
    // CHECK-SAME: (%[[DEV_REAL_I32:.+]]: i32, %[[DEV_IMAG_I32:.+]]: i32, %arg2: !stream.binding)
    func.func @device_complex_f32_bitcast(%arg0: complex<f32>, %arg1: !stream.binding) {
      // CHECK-DAG: %[[DEV_REAL_F32:.+]] = arith.bitcast %[[DEV_REAL_I32]] : i32 to f32
      // CHECK-DAG: %[[DEV_IMAG_F32:.+]] = arith.bitcast %[[DEV_IMAG_I32]] : i32 to f32
      // CHECK-DAG: %[[DEV_COMPLEX:.+]] = complex.create %[[DEV_REAL_F32]], %[[DEV_IMAG_F32]]
      // CHECK-NEXT: util.optimization_barrier %[[DEV_COMPLEX]]
      util.optimization_barrier %arg0 : complex<f32>
      return
    }
  }
}
// CHECK-LABEL: func.func @host_complex_bitcast
func.func @host_complex_bitcast(%arg0: complex<f32>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}
  // CHECK-DAG: %[[HOST_REAL_F32:.+]] = complex.re %arg0
  // CHECK-DAG: %[[HOST_IMAG_F32:.+]] = complex.im %arg0
  // CHECK-DAG: %[[HOST_REAL_I32:.+]] = arith.bitcast %[[HOST_REAL_F32]] : f32 to i32
  // CHECK-DAG: %[[HOST_IMAG_I32:.+]] = arith.bitcast %[[HOST_IMAG_F32]] : f32 to i32
  %1 = complex.bitcast %arg0 : complex<f32> to i64
  %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c128}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[HOST_REAL_I32]], %[[HOST_IMAG_I32]] : i32, i32)
    stream.cmd.dispatch @ex5::@device_complex_f32_bitcast[%c1, %c1, %c1](%1 : i64) {
      wo %arg1[%c0 for %c128] : !stream.resource<external>{%c128}
    }
  } => !stream.timepoint
  return %2 : !stream.timepoint
}
