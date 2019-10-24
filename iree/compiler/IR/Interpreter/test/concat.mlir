// RUN: iree-opt -pass-pipeline='func(canonicalize)' %s --split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @concat.1D
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.1D(%arg0 : memref<4xi32>, %arg1 : memref<3xi32>) -> memref<7xi32> {
  // CHECK-DAG: [[SRC_INDICES:%.+]]  = iree.constant[dense<0> : tensor<1x
  // CHECK-DAG: [[DST:%.+]]          = "iree_hl_interp.alloc_heap"() : () -> memref<7xi32>

  // CHECK-DAG: [[DST_INDICES0:%.+]] = iree.constant[dense<0> : tensor<1x
  // CHECK-DAG: [[LENGTHS0:%.+]]     = iree.constant[dense<4> : tensor<1x
  // CHECK-DAG: "iree_hl_interp.copy"([[ARG0]], [[SRC_INDICES]], [[DST]], [[DST_INDICES0]], [[LENGTHS0]])

  // CHECK-DAG: [[DST_INDICES1:%.+]] = iree.constant[dense<4> : tensor<1x
  // CHECK-DAG: [[LENGTHS1:%.+]]     = iree.constant[dense<3> : tensor<1x
  // CHECK-DAG: "iree_hl_interp.copy"([[ARG1]], [[SRC_INDICES]], [[DST]], [[DST_INDICES1]], [[LENGTHS1]])

  %0 = "iree_hl_interp.concat"(%arg0, %arg1) {dimension = 0 : i32} : (memref<4xi32>, memref<3xi32>) -> memref<7xi32>

  // CHECK: return [[DST]]
  return %0 : memref<7xi32>
}

// -----

// CHECK-LABEL: func @concat.2D.Dim0
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.2D.Dim0(%arg0 : memref<4x4xi32>, %arg1 : memref<3x4xi32>) -> memref<7x4xi32> {
  // CHECK-DAG: [[SRC_INDICES:%.+]]  = iree.constant[dense<0> : tensor<2x
  // CHECK-DAG: [[DST:%.+]]          = "iree_hl_interp.alloc_heap"() : () -> memref<7x4xi32>

  // CHECK-DAG: [[DST_INDICES0:%.+]] = iree.constant[dense<0> : tensor<2x
  // CHECK-DAG: [[LENGTHS0:%.+]]     = iree.constant[dense<4> : tensor<2x
  // CHECK-DAG: "iree_hl_interp.copy"([[ARG0]], [[SRC_INDICES]], [[DST]], [[DST_INDICES0]], [[LENGTHS0]])

  // CHECK-DAG: [[DST_INDICES1:%.+]] = iree.constant[dense<[4, 0]>
  // CHECK-DAG: [[LENGTHS1:%.+]]     = iree.constant[dense<[3, 4]>
  // CHECK-DAG: "iree_hl_interp.copy"([[ARG1]], [[SRC_INDICES]], [[DST]], [[DST_INDICES1]], [[LENGTHS1]])

  %0 = "iree_hl_interp.concat"(%arg0, %arg1) {dimension = 0 : i32} : (memref<4x4xi32>, memref<3x4xi32>) -> memref<7x4xi32>

  // CHECK: return [[DST]]
  return %0 : memref<7x4xi32>
}

// -----

// CHECK-LABEL: func @concat.2D.Dim1
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.2D.Dim1(%arg0 : memref<4x4xi32>, %arg1 : memref<4x3xi32>) -> memref<4x7xi32> {
  // CHECK-DAG: [[SRC_INDICES:%.+]]  = iree.constant[dense<0> : tensor<2x
  // CHECK-DAG: [[DST:%.+]]          = "iree_hl_interp.alloc_heap"() : () -> memref<4x7xi32>

  // CHECK-DAG: [[DST_INDICES0:%.+]] = iree.constant[dense<0> : tensor<2x
  // CHECK-DAG: [[LENGTHS0:%.+]]     = iree.constant[dense<4> : tensor<2x
  // CHECK-DAG: "iree_hl_interp.copy"([[ARG0]], [[SRC_INDICES]], [[DST]], [[DST_INDICES0]], [[LENGTHS0]])

  // CHECK-DAG: [[DST_INDICES1:%.+]] = iree.constant[dense<[0, 4]>
  // CHECK-DAG: [[LENGTHS1:%.+]]     = iree.constant[dense<[4, 3]>
  // CHECK-DAG: "iree_hl_interp.copy"([[ARG1]], [[SRC_INDICES]], [[DST]], [[DST_INDICES1]], [[LENGTHS1]])

  %0 = "iree_hl_interp.concat"(%arg0, %arg1) {dimension = 1 : i32} : (memref<4x4xi32>, memref<4x3xi32>) -> memref<4x7xi32>

  // CHECK: return [[DST]]
  return %0 : memref<4x7xi32>
}
