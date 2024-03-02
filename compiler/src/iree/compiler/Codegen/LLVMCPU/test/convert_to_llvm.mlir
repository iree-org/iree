// RUN: iree-opt -iree-convert-to-llvm --split-input-file %s | FileCheck %s

builtin.module {
  func.func private @extern_public()
  func.func @entry_point() {
    return
  }
}
//      CHECK: llvm.func @extern_public()
//      CHECK: llvm.func @entry_point(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias},
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias},
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias}) -> i32
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

module {
  func.func @interleave_and_bitcast_lowering() {
    %cst = arith.constant dense<4> : vector<4x2xi8>
    %cst_0 = arith.constant dense<0> : vector<4x4xi4>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4096 = arith.constant 4096 : index
    %c8192 = arith.constant 8192 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c4096) flags(ReadOnly) : memref<128xi8, strided<[1], offset: 4096>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c8192) : memref<256x64xi8, strided<[64, 1], offset: 8192>>
    %2 = vector.load %0[%c0] : memref<128xi8, strided<[1], offset: 4096>>, vector<2xi8>
    %3 = vector.bitcast %2 : vector<2xi8> to vector<4xi4>
    %4 = vector.insert %3, %cst_0 [3] : vector<4xi4> into vector<4x4xi4>
    %5 = vector.bitcast %4 : vector<4x4xi4> to vector<4x2xi8>
    %6 = arith.shli %5, %cst : vector<4x2xi8>
    %7 = arith.shrsi %6, %cst : vector<4x2xi8>
    %8 = arith.shrsi %5, %cst : vector<4x2xi8>
    %9 = vector.interleave %7, %8 : vector<4x2xi8>
    %10 = vector.extract %9[0] : vector<4xi8> from vector<4x4xi8>
    %11 = vector.extract %9[1] : vector<4xi8> from vector<4x4xi8>
    %12 = vector.extract %9[2] : vector<4xi8> from vector<4x4xi8>
    %13 = vector.extract %9[3] : vector<4xi8> from vector<4x4xi8>
    vector.store %10, %1[%c0, %c0] : memref<256x64xi8, strided<[64, 1], offset: 8192>>, vector<4xi8>
    vector.store %11, %1[%c1, %c0] : memref<256x64xi8, strided<[64, 1], offset: 8192>>, vector<4xi8>
    vector.store %12, %1[%c2, %c0] : memref<256x64xi8, strided<[64, 1], offset: 8192>>, vector<4xi8>
    vector.store %13, %1[%c3, %c0] : memref<256x64xi8, strided<[64, 1], offset: 8192>>, vector<4xi8>
    return
  }
}

// Make sure we can lowering multi-dimensional `vector.interleave` and its
// corresponding multi-dimensional `vector.bitcast`.

// CHECK-LABEL: llvm.func @interleave_and_bitcast_lowering(
//   CHECK-NOT:   vector.bitcast %{{.*}} : vector<4x4xi4> to vector<4x2xi8>
//   CHECK-NOT:   vector.interleave

