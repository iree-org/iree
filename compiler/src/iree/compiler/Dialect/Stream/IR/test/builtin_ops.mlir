// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @builtinSplatI64
func.func @builtinSplatI64(%arg0: index, %arg1: i64) -> !stream.resource<*> {
  // CHECK: = stream.builtin.splat.i64 %arg1 : i64 -> !stream.resource<*>{%arg0}
  %0 = stream.builtin.splat.i64 %arg1 : i64 -> !stream.resource<*>{%arg0}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @builtinFillI64
func.func @builtinFillI64(%arg0: !stream.resource<*>, %arg1: index, %arg2: i64) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.builtin.fill.i64 %arg2, %arg0[%c0 to %c128 for %c128] : i64 -> %arg0 as !stream.resource<*>{%arg1}
  %0 = stream.builtin.fill.i64 %arg2, %arg0[%c0 to %c128 for %c128] : i64 -> %arg0 as !stream.resource<*>{%arg1}
  return %0 : !stream.resource<*>
}
