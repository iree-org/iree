// RUN: iree-opt %s -iree-interpreter-load-store-data-flow-opt -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @scalarLoadStore
// CHECK-SAME: [[SRC:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[DST:%[a-zA-Z0-9]+]]
func @scalarLoadStore(%src: memref<f32>, %dst: memref<f32>) {
  // CHECK-DAG: [[EMPTY_MEMREF:%.+]] = iree.constant dense<[]> : tensor<0xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[EMPTY_MEMREF]], [[DST]], [[EMPTY_MEMREF]], [[EMPTY_MEMREF]])
  %0 = load %src[] : memref<f32>
  store %0, %dst[] : memref<f32>
  // CHECK-NEXT: return
  return
}
