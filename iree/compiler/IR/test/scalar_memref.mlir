// RUN: iree-opt %s -canonicalize | FileCheck %s --dump-input=fail

// CHECK-LABEL: @fold_memref_to_memref
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_memref_to_memref(%arg0 : memref<i32>) -> memref<i32> {
  // CHECK-NEXT: return [[ARG]]
  %0 = iree.memref_to_scalar(%arg0 : memref<i32>) : i32
  %1 = iree.scalar_to_memref(%0 : i32) : memref<i32>
  return %1 : memref<i32>
}

// CHECK-LABEL: @fold_scalar_to_scalar
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_scalar_to_scalar(%arg0 : i32) -> i32 {
  // CHECK-NEXT: return [[ARG]]
  %0 = iree.scalar_to_memref(%arg0 : i32) : memref<i32>
  %1 = iree.memref_to_scalar(%0 : memref<i32>) : i32
  return %1 : i32
}
