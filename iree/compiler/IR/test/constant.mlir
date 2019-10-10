// RUN: iree-opt %s | iree-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: @const
func @const() -> (memref<i32>, memref<i32>, memref<i32>, memref<i32>) {
  // CHECK: iree.constant[dense<1> : tensor<i32>] : memref<i32>
  %0 = iree.constant[dense<1> : tensor<i32>] : memref<i32>
  // CHECK-NEXT: iree.constant[dense<1> : tensor<i32>] : memref<i32>
  %1 = "iree.constant"() {value = dense<1> : tensor<i32>} : () -> memref<i32>
  // CHECK-NEXT: iree.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  %2 = "iree.constant"() {attr = "foo", value = dense<1> : tensor<i32>} : () -> memref<i32>
  // CHECK-NEXT: iree.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  %3 = iree.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  return %0, %1, %2, %3 : memref<i32>, memref<i32>, memref<i32>, memref<i32>
}
