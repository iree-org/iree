// RUN: iree-opt %s -pass-pipeline='func(canonicalize)' -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @necessary_clone_not_removed
func @necessary_clone_not_removed() -> (memref<i32>, memref<i32>) {
  // CHECK: [[ORIG:%.+]] = iree.constant[dense<1> : tensor<i32>]
  %original = iree.constant[dense<1> : tensor<i32>] : memref<i32>
  // CHECK: [[CLONE:%.+]] = "iree_hl_interp.clone"([[ORIG]])
  %cloned = "iree_hl_interp.clone"(%original) : (memref<i32>) -> memref<i32>
  %other = iree.constant[dense<2> : tensor<i32>] : memref<i32>
  %empty = iree.constant[dense<[]> : tensor<0xi32>] : memref<0xi32>
  "iree_hl_interp.copy"(%other, %empty, %original, %empty, %empty) : (memref<i32>, memref<0xi32>, memref<i32>, memref<0xi32>, memref<0xi32>) -> ()
  // CHECK: return [[CLONE]], [[ORIG]]
  return %cloned, %original : memref<i32>, memref<i32>
}

// -----

// CHECK-LABEL: @unnecessary_clone_removed
func @unnecessary_clone_removed() -> memref<i32> {
  // CHECK: [[ORIG:%.+]] = iree.constant[dense<1> : tensor<i32>]
  %original = iree.constant[dense<1> : tensor<i32>] : memref<i32>
  %cloned = "iree_hl_interp.clone"(%original) : (memref<i32>) -> memref<i32>
  // CHECK: return [[ORIG]]
  return %cloned : memref<i32>
}
