// RUN: iree-opt %s -iree-interpreter-load-store-data-flow-opt -split-input-file | FileCheck %s --dump-input=fail

// NOTE: There are no check statements for the constants created for specifying
// indices to store into because matching those with FileCheck is more trouble
// than its worth and anything other than a constant index 0 passed to the store
// would error in MLIR verification anyway.

// CHECK-LABEL: func @scalarLoadStore
// CHECK-SAME: [[SRC:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[DST:%[a-zA-Z0-9]+]]
func @scalarLoadStore(%src: memref<f32>, %dst: memref<f32>) {
  // CHECK-DAG: [[SRC_INDICES:%.+]] = iree.constant 0 : index
  %0 = load %src[] : memref<f32>
  // CHECK-DAG: [[DST_INDICES:%.+]] = iree.constant 0 : index
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant 1 : index
  store %0, %dst[] : memref<f32>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @replacementLoadStore
// CHECK-SAME: [[SRC:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[DST:%[a-zA-Z0-9]+]]
func @replacementLoadStore(%src: memref<1xf32>, %dst: memref<1xf32>) {
  // CHECK: [[C0:%.+]] = constant 0 : index
  %c0 = constant 0 : index
  // CHECK-DAG: [[SRC_INDICES:%.+]] = alloc() : memref<1xindex>
  // Not checked: constant created for store index
  // CHECK-DAG: store [[C0]], [[SRC_INDICES]]
  %0 = load %src[%c0] : memref<1xf32>
  // CHECK-DAG: [[DST_INDICES:%.+]] = alloc() : memref<1xindex>
  // Not checked: constant created for store index
  // CHECK-DAG: store [[C0]], [[DST_INDICES]]
  store %0, %dst[%c0] : memref<1xf32>
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant 1 : index
  // CHECK: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @offsetLoad
// CHECK-SAME: [[SRC:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[DST:%[a-zA-Z0-9]+]]
func @offsetLoad(%src: memref<4xf32>, %dst: memref<f32>) {
  // CHECK: [[C1:%.+]] = constant 1 : index
  %c1 = constant 1 : index
  // CHECK-DAG: [[SRC_INDICES:%.+]] = alloc() : memref<1xindex>
  // CHECK-DAG:  store [[C1]], [[SRC_INDICES]]
  %1 = load %src[%c1] : memref<4xf32>
  // CHECK-DAG: [[DST_INDICES:%.+]] = iree.constant 0 : index
  store %1, %dst[] : memref<f32>
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant 1 : index
  // CHECK: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @offsetStore
// CHECK-SAME: [[SRC:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[DST:%[a-zA-Z0-9]+]]
func @offsetStore(%src: memref<f32>, %dst: memref<4xf32>) {
  // CHECK-DAG: [[SRC_INDICES:%.+]] = iree.constant 0 : index
  %0 = load %src[] : memref<f32>
  // CHECK-DAG: [[C2:%.+]] = constant 2 : index
  %c2 = constant 2 : index
  // CHECK-DAG: [[DST_INDICES:%.+]] = alloc() : memref<1xindex>
  // Not checked: constant created for store index
  // CHECK-DAG: store [[C2]], [[DST_INDICES]]
  store %0, %dst[%c2] : memref<4xf32>
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant 1 : index
  // CHECK: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @offsetLoadStore
// CHECK-SAME: [[SRC:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[DST:%[a-zA-Z0-9]+]]
func @offsetLoadStore(%src: memref<4xf32>, %dst: memref<4xf32>) {
  // CHECK-DAG: [[C1:%.+]] = constant 1 : index
  %c1 = constant 1 : index
  // CHECK-DAG: [[SRC_INDICES:%.+]] = alloc() : memref<1xindex>
  // Not checked: constant created for store index
  // CHECK-DAG: store [[C1]], [[SRC_INDICES]]
  %1 = load %src[%c1] : memref<4xf32>
  // CHECK-DAG: [[C2:%.+]] = constant 2 : index
  %c2 = constant 2 : index
  // CHECK-DAG: [[DST_INDICES:%.+]] = alloc() : memref<1xindex>
  // Not checked: constant created for store index
  // CHECK-DAG: store [[C2]], [[DST_INDICES]]
  store %1, %dst[%c2] : memref<4xf32>
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant 1 : index
  // CHECK: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @offsetLoadStoreSameIndices
// CHECK-SAME: [[SRC:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[DST:%[a-zA-Z0-9]+]]
func @offsetLoadStoreSameIndices(%src: memref<4xf32>, %dst: memref<4xf32>) {
  // CHECK-DAG: [[C1:%.+]] = constant 1 : index
  %c1 = constant 1 : index
  // CHECK-DAG: [[SRC_INDICES:%.+]] = alloc() : memref<1xindex>
  // Not checked: constant created for store index
  // CHECK-DAG: store [[C1]], [[SRC_INDICES]]
  %1 = load %src[%c1] : memref<4xf32>
  // CHECK-DAG: [[DST_INDICES:%.+]] = alloc() : memref<1xindex>
  // Not checked: constant created for store index
  // CHECK-DAG: store [[C1]], [[DST_INDICES]]
  store %1, %dst[%c1] : memref<4xf32>
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant 1 : index
  // CHECK: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: return
  return
}
