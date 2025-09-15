// RUN: iree-opt -iree-convert-to-llvm --split-input-file %s | FileCheck %s

builtin.module {
  func.func private @extern_public()
  func.func @entry_point() {
    return
  }
}
//      CHECK: llvm.func @extern_public()
//      CHECK: llvm.func @entry_point(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef},
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef},
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32
//      CHECK:     llvm.return %{{.+}} : i32

// -----

module {
  func.func private @default_cconv_with_extra_fields(memref<f32>, i32, f64) -> (f32) attributes {
      hal.import.bitcode = true,
      hal.import.cconv = 0 : i32,
      hal.import.fields = ["processor_data", "processor_id"],
      llvm.bareptr = true
  }
  func.func @bar() {
    %c0 = arith.constant 42 : i32
    %c1 = arith.constant 42.0 : f64
    %0 = memref.alloca() : memref<f32>
    %1 = call @default_cconv_with_extra_fields(%0, %c0, %c1) : (memref<f32>, i32, f64) -> (f32)
    return
  }
}
//      CHECK: llvm.func @default_cconv_with_extra_fields(!llvm.ptr, i32, f64, !llvm.ptr, i32) -> f32
//      CHECK: llvm.func @bar
//  CHECK-DAG:   %[[Ci32:.+]] = llvm.mlir.constant(42 : i32) : i32
//  CHECK-DAG:   %[[Cf64:.+]] = llvm.mlir.constant(4.200000e+01 : f64) : f64
//  CHECK-DAG:   %[[ALLOCA:.+]] = llvm.alloca
//  CHECK-DAG:   %[[DATA:.+]] = llvm.getelementptr inbounds %arg0[4]
//  CHECK-DAG:   %[[PROCESSOR_INFO:.+]] = llvm.load %arg2
//      CHECK:   %[[PROCESSOR_ID:.+]] = llvm.extractvalue %[[PROCESSOR_INFO]][4]
//      CHECK: %[[VAL:.+]] = llvm.call @default_cconv_with_extra_fields
// CHECK-SAME:     (%[[ALLOCA]], %[[Ci32]], %[[Cf64]], %[[DATA]], %[[PROCESSOR_ID]])

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @interleave_and_bitcast_lowering() {
  %cst = arith.constant dense<4> : vector<4x2xi8>
  %cst_0 = arith.constant dense<0> : vector<4x4xi4>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4096 = arith.constant 4096 : index
  %c8192 = arith.constant 8192 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c4096) flags(ReadOnly) : memref<128xi8, strided<[1], offset: ?>>
  %out_buffer = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c8192) : memref<256x64xi4, strided<[64, 1], offset: 8192>>
  %2 = vector.load %0[%c0] : memref<128xi8, strided<[1], offset: ?>>, vector<2xi8>
  %3 = vector.bitcast %2 : vector<2xi8> to vector<4xi4>
  %4 = vector.insert %3, %cst_0 [3] : vector<4xi4> into vector<4x4xi4>
  %5 = vector.bitcast %4 : vector<4x4xi4> to vector<4x2xi8>
  %6 = arith.shli %5, %cst : vector<4x2xi8>
  %7 = arith.shrsi %6, %cst : vector<4x2xi8>
  %8 = arith.shrsi %5, %cst : vector<4x2xi8>

  // Ops that should be lowered
  %9 = vector.interleave %7, %8 : vector<4x2xi8> -> vector<4x4xi8>
  %14 = vector.bitcast %9 : vector<4x4xi8> to vector<4x8xi4>

  vector.store %14, %out_buffer[%c0, %c0] : memref<256x64xi4, strided<[64, 1], offset: 8192>>, vector<4x8xi4>
  return
}

// Make sure we can lower multi-dimensional `vector.interleave` and its
// corresponding multi-dimensional `vector.bitcast`.

// CHECK-LABEL: llvm.func @interleave_and_bitcast_lowering(
// vector.interleave should be gone entirely
//   CHECK-NOT:   vector.interleave
// 2D vector.bitcast tha followed should be replaced with 1D vector.bitcast
//       CHECK:   llvm.bitcast {{.*}} : vector<4xi8> to vector<8xi4>
//   CHECK-NOT:   vector.bitcast %{{.*}} : vector<4x4xi8> to vector<4x8xi4>

// -----

module attributes {
    hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple="riscv64-unknown-elf"}>
} {
  func.func private @gather_lowering(%buffer : memref<2048xf32>, %index : vector<64xindex>, %mask : vector<64xi1>) -> vector<64xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %r = vector.gather %buffer[%c0] [%index], %mask, %cst : memref<2048xf32>, vector<64xindex>, vector<64xi1>, vector<64xf32> into vector<64xf32>
    return %r : vector<64xf32>
  }
}

// CHECK-LABEL:   llvm.func @gather_lowering(
// CHECK-NOT: llvm.intr.masked.gather

// -----

module attributes {
    hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple="riscv64-unknown-elf", cpu_features ="+v"}>
} {
  func.func private @negative_no_gather_lowering(%buffer : memref<2048xf32>, %index : vector<64xindex>, %mask : vector<64xi1>) -> vector<64xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %r = vector.gather %buffer[%c0] [%index], %mask, %cst : memref<2048xf32>, vector<64xindex>, vector<64xi1>, vector<64xf32> into vector<64xf32>
    return %r : vector<64xf32>
  }
}

// CHECK-LABEL:   llvm.func @negative_no_gather_lowering(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<64xi64>,
// CHECK-SAME:                                           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: vector<64xi1>) -> vector<64xf32> attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<64xf32>) : vector<64xf32>
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_5]]] : (!llvm.ptr, vector<64xi64>) -> vector<64x!llvm.ptr>, f32
// CHECK:           %[[VAL_9:.*]] = llvm.intr.masked.gather %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] {alignment = 4 : i32} : (vector<64x!llvm.ptr>, vector<64xi1>, vector<64xf32>) -> vector<64xf32>
// CHECK:           llvm.return %[[VAL_9]] : vector<64xf32>
// CHECK:         }
