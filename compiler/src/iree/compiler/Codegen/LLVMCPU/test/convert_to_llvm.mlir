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
