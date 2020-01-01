// RUN: iree-opt %s | iree-opt | IreeFileCheck %s

// CHECK-LABEL: @const
func @const() -> (memref<i32>, memref<i32>, memref<i32>, memref<i32>) {
  // CHECK: iree_interp.constant[dense<1> : tensor<i32>] : memref<i32>
  %0 = iree_interp.constant[dense<1> : tensor<i32>] : memref<i32>
  // CHECK-NEXT: iree_interp.constant[dense<1> : tensor<i32>] : memref<i32>
  %1 = "iree_interp.constant"() {value = dense<1> : tensor<i32>} : () -> memref<i32>
  // CHECK-NEXT: iree_interp.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  %2 = "iree_interp.constant"() {attr = "foo", value = dense<1> : tensor<i32>} : () -> memref<i32>
  // CHECK-NEXT: iree_interp.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  %3 = iree_interp.constant[dense<1> : tensor<i32>] {attr = "foo"} : memref<i32>
  return %0, %1, %2, %3 : memref<i32>, memref<i32>, memref<i32>, memref<i32>
}
