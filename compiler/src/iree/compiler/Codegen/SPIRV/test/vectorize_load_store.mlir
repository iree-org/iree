// RUN: iree-opt --split-input-file --iree-spirv-vectorize-load-store --canonicalize -cse --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: func.func @alloc_copy
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x1024xvector<4xf32>>, %[[X:.+]]: index, %[[Y:.+]]: index)
//       CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[IDX:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 4)>()[%[[Y]]]
//       CHECK: %[[V:.+]] = memref.load %[[ARG0]][%[[X]], %[[IDX]]] : memref<4096x1024xvector<4xf32>>
//       CHECK: memref.store %[[V]], %[[ALLOC]][%[[X]], %[[IDX]]] : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[MAT:.+]] = vector.transfer_read %[[ARG0]][%[[X]], %[[IDX]]], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//       CHECK: vector.transfer_write %[[MAT]], %[[ALLOC]][%[[X]], %[[IDX]]] : vector<32x8xf32>, memref<128x8xvector<4xf32>, 3>
//       CHECK: memref.dealloc %[[ALLOC]] : memref<128x8xvector<4xf32>, 3>
func.func @alloc_copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<128x32xf32, 3>
  %v = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<1x4xf32>
  vector.transfer_write %v, %0[%x, %y] : vector<1x4xf32>, memref<128x32xf32, 3>
  %mat = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %0[%x, %y] : vector<32x8xf32>, memref<128x32xf32, 3>
  memref.dealloc %0 : memref<128x32xf32, 3>
  return
}

// -----

// Test that the memref is not vectorized if used by scalar load or store.

// CHECK-LABEL: func.func @alloc_copy
//  CHECK-SAME: %[[ARG0:.+]]: memref<4096x4096xf32>
func.func @alloc_copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<128x32xf32, 3>
  %s = memref.load %arg0[%x, %y] : memref<4096x4096xf32>
  memref.store %s, %0[%x, %y] : memref<128x32xf32, 3>
  memref.dealloc %0 : memref<128x32xf32, 3>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy()
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf32>, memref<4096x1024xvector<4xf32>>
func.func @resource_copy() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x4096xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf32>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<1x4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf32>, memref<4096x4096xf32>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_with_offset()
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%{{.*}}) : memref<2048x4096x1024xvector<4xf32>, strided<[4194304, 1024, 1], offset: ?>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<2048x4096x1024xvector<4xf32>, strided<[4194304, 1024, 1], offset: ?>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<2048x4096x1024xvector<4xf32>, strided<[4194304, 1024, 1], offset: ?>>, vector<32x8xf32>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf32>, memref<4096x1024xvector<4xf32>>
func.func @resource_copy_with_offset() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %offset = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<2048x4096x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf32>
  %v = vector.transfer_read %0[%c0, %c0, %c0], %cst : memref<2048x4096x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, vector<1x4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf32>, memref<4096x4096xf32>
  %mat = vector.transfer_read %0[%c0, %c0, %c0], %cst : memref<2048x4096x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_f16
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf16>>, vector<32x8xf16>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf16>, memref<4096x1024xvector<4xf16>>
func.func @resource_copy_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x4096xf16>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<1x4xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf16>, memref<4096x4096xf16>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<32x8xf16>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf16>, memref<4096x4096xf16>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_8xf16
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x512xvector<4xf32>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x512xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x512xvector<4xf32>>, vector<32x8xf16>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf16>, memref<4096x512xvector<4xf32>>
func.func @resource_copy_8xf16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x4096xf16>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<1x8xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x8xf16>, memref<4096x4096xf16>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<32x8xf16>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf16>, memref<4096x4096xf16>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_dynamic_shape()
func.func @resource_copy_dynamic_shape() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM0:.+]] = hal.interface.constant.load[0] : index
  // CHECK: %[[DIM1:.+]] = hal.interface.constant.load[1] : index
  %dim0 = hal.interface.constant.load[0] : index
  %dim1 = hal.interface.constant.load[1] : index

  // CHECK: %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x8x?x32xvector<4xf32>>{%[[DIM0]], %[[DIM1]]}
  // CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x8x?x32xvector<4xf32>>{%[[DIM0]], %[[DIM1]]}
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x8x?x128xf32>{%dim0, %dim1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x8x?x128xf32>{%dim0, %dim1}

  // CHECK: %[[VAL1:.+]] = memref.load %[[INPUT]]
  // CHECK: memref.store %[[VAL1]], %[[OUTPUT]]
  %v = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst : memref<?x8x?x128xf32>, vector<1x4xf32>
  vector.transfer_write %v, %1[%c0, %c0, %c0, %c0] : vector<1x4xf32>, memref<?x8x?x128xf32>

  // CHECK: %[[VAL2:.+]] = vector.transfer_read %[[INPUT]]
  // CHECK: vector.transfer_write %[[VAL2]], %[[OUTPUT]]
  %mat = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst : memref<?x8x?x128xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0, %c0, %c0] : vector<32x8xf32>, memref<?x8x?x128xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_dynamic_last_dim()
func.func @resource_copy_dynamic_last_dim() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %dim = hal.interface.constant.load[0] : index
  // CHECK: hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x?xf32>
  // CHECK: hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x?xf32>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x?xf32>{%dim}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x?xf32>{%dim}
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x?xf32>, vector<1x4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf32>, memref<4096x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @do_not_vectorize_odd_vector_size
func.func @do_not_vectorize_odd_vector_size() {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: hal.interface.binding.subspan
  // CHECK-SAME: memref<4x3xf32>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x3xf32>
  // CHECK: hal.interface.binding.subspan
  // CHECK-SAME: memref<4x3xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4x3xf32>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4x3xf32>, vector<3xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<3xf32>, memref<4x3xf32>
  return
}

// -----

func.func @vectorize_binding_subspan() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
  // CHECK: hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x4096xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf32>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
  return
}


// -----

// CHECK-LABEL: func.func @scalarize_vector_transfer_op
func.func @scalarize_vector_transfer_op(%arg: vector<3xf32>) -> (vector<3xf32>) {
  %c0 = arith.constant 0: index
  %c3 = arith.constant 3: index
  %f0 = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<20xf32>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<20xf32>
  // CHECK-DAG: %[[INDEX0:.+]] = arith.constant 3 : index
  // CHECK-DAG: %[[INDEX1:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[INDEX2:.+]] = arith.constant 5 : index
  // CHECK-DAG: %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<3xf32>

  // CHECK: %[[ELEM0:.+]] = memref.load %{{.+}}[%[[INDEX0]]]
  // CHECK: %[[V0:.+]] = vector.insert %[[ELEM0]], %[[CST]] [0] : f32 into vector<3xf32>
  // CHECK: %[[ELEM1:.+]] = memref.load %{{.+}}[%[[INDEX1]]]
  // CHECK: %[[V1:.+]] = vector.insert %[[ELEM1]], %[[V0]] [1] : f32 into vector<3xf32>
  // CHECK: %[[ELEM2:.+]] = memref.load %{{.+}}[%[[INDEX2]]]
  // CHECK: %[[V2:.+]] = vector.insert %[[ELEM2]], %[[V1]] [2] : f32 into vector<3xf32>
  // CHECK: %[[EXT_0:.+]] = vector.extract %{{.*}}[0] : vector<3xf32>
  // CHECK: memref.store %[[EXT_0]], %{{.*}}[%[[INDEX0]]] : memref<20xf32>
  // CHECK: %[[EXT_1:.+]] = vector.extract %{{.*}}[1] : vector<3xf32>
  // CHECK: memref.store %[[EXT_1]], %{{.*}}[%[[INDEX1]]] : memref<20xf32>
  // CHECK: %[[EXT_2:.+]] = vector.extract %{{.*}}[2] : vector<3xf32>
  // CHECK: memref.store %[[EXT_2]], %{{.*}}[%[[INDEX2]]] : memref<20xf32>
  // CHECK: return %[[V2]] : vector<3xf32>
  %3 = vector.transfer_read %0[%c3], %f0 : memref<20xf32>, vector<3xf32>
  vector.transfer_write %arg, %2[%c3] : vector<3xf32>, memref<20xf32>
  return %3: vector<3xf32>
}

// -----

// CHECK-LABEL: func.func @scalarize_non_minor_identity_transfer_read
//  CHECK-SAME: (%[[MEM:.+]]: memref<4x2x4xi32>, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
func.func @scalarize_non_minor_identity_transfer_read(%memory: memref<4x2x4xi32>, %i1: index, %i2: index, %i3: index) -> vector<4xi32> {
  %c0 = arith.constant 0 : i32
  %0 = vector.transfer_read %memory[%i1, %i2, %i3], %c0 {
    in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d0)>
  } : memref<4x2x4xi32>, vector<4xi32>
  return %0: vector<4xi32>
}

// CHECK: %[[INIT:.+]] = arith.constant dense<0> : vector<4xi32>
// CHECK: %[[LD0:.+]] = memref.load %[[MEM]][%[[I1]], %[[I2]], %[[I3]]] : memref<4x2x4xi32>
// CHECK: %[[INSERT0:.+]] = vector.insert %[[LD0]], %[[INIT]] [0] : i32 into vector<4xi32>
// CHECK: %[[IDX1:.+]] = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%[[I1]]]
// CHECK: %[[LD1:.+]] = memref.load %[[MEM]][%[[IDX1]], %[[I2]], %[[I3]]] : memref<4x2x4xi32>
// CHECK: %[[INSERT1:.+]] = vector.insert %[[LD1]], %[[INSERT0]] [1] : i32 into vector<4xi32>
// CHECK: %[[IDX2:.+]] = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%[[I1]]]
// CHECK: %[[LD2:.+]] = memref.load %[[MEM]][%[[IDX2]], %[[I2]], %[[I3]]] : memref<4x2x4xi32>
// CHECK: %[[INSERT2:.+]] = vector.insert %[[LD2]], %[[INSERT1]] [2] : i32 into vector<4xi32>
// CHECK: %[[IDX3:.+]] = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%[[I1]]]
// CHECK: %[[LD3:.+]] = memref.load %[[MEM]][%[[IDX3]], %[[I2]], %[[I3]]] : memref<4x2x4xi32>
// CHECK: %[[INSERT3:.+]] = vector.insert %[[LD3]], %[[INSERT2]] [3] : i32 into vector<4xi32>
// CHECK: return %[[INSERT3]]

// -----

// CHECK-LABEL: func.func @scalarize_non_minor_identity_transfer_write
//  CHECK-SAME: (%[[VALUE:.+]]: vector<4xf32>, %[[I1:.+]]: index, %[[I2:.+]]: index)
func.func @scalarize_non_minor_identity_transfer_write(%value: vector<4xf32>, %i1: index, %i2: index) {
  %c0 = arith.constant 0: index
  %buffer = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<1x130x130x64xf32>
  vector.transfer_write %value, %buffer[%c0, %i1, %i2, %c0] {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>} : vector<4xf32>, memref<1x130x130x64xf32>
  return
}

// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[BUFFER:.+]] = hal.interface.binding.subspan
// CHECK: %[[E0:.+]] = vector.extract %[[VALUE]][0] : vector<4xf32>
// CHECK: memref.store %[[E0]], %[[BUFFER]][%[[C0]], %[[I1]], %[[I2]], %[[C0]]]
// CHECK: %[[PLUS1:.+]] = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%[[I2]]]
// CHECK: %[[E1:.+]] = vector.extract %[[VALUE]][1] : vector<4xf32>
// CHECK: memref.store %[[E1]], %[[BUFFER]][%[[C0]], %[[I1]], %[[PLUS1]], %[[C0]]]
// CHECK: %[[PLUS2:.+]] = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%[[I2]]]
// CHECK: %[[E2:.+]] = vector.extract %[[VALUE]][2] : vector<4xf32>
// CHECK: memref.store %[[E2]], %[[BUFFER]][%[[C0]], %[[I1]], %[[PLUS2]], %[[C0]]]
// CHECK: %[[PLUS3:.+]] = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%[[I2]]]
// CHECK: %[[E3:.+]] = vector.extract %[[VALUE]][3] : vector<4xf32>
// CHECK: memref.store %[[E3]], %[[BUFFER]][%[[C0]], %[[I1]], %[[PLUS3]], %[[C0]]]

// -----

// CHECK-LABEL: func.func @scalarize_0d_transfer_read
//  CHECK-SAME: (%[[MEM:.+]]: memref<4xf32>, %[[I:.+]]: index)
func.func @scalarize_0d_transfer_read(%memory: memref<4xf32>, %i: index) -> vector<f32> {
  %f0 = arith.constant 0.0 : f32
  %v = vector.transfer_read %memory[%i], %f0 : memref<4xf32>, vector<f32>
  return %v : vector<f32>
}

// CHECK: %[[S:.+]] = memref.load %[[MEM]][%[[I]]] : memref<4xf32>
// CHECK: %[[V:.+]] = vector.splat %[[S]] : vector<f32>
// CHECK: return %[[V]]

// -----

// CHECK-LABEL: func.func @scalarize_0d_transfer_write
//  CHECK-SAME: (%[[V:.+]]: vector<f32>, %[[MEM:.+]]: memref<4xf32>, %[[I:.+]]: index)
func.func @scalarize_0d_transfer_write(%val: vector<f32>, %memory: memref<4xf32>, %i: index) {
  vector.transfer_write %val, %memory[%i] : vector<f32>, memref<4xf32>
  return
}

// CHECK: %[[S:.+]] = vector.extractelement %[[V]][] : vector<f32>
// CHECK: memref.store %[[S]], %[[MEM]][%[[I]]] : memref<4xf32>

// -----

// CHECK-LABEL: func.func @scalarize_indivisible_vector_transfer_read_op
func.func @scalarize_indivisible_vector_transfer_read_op(%i: index) -> vector<4xf32> {
  %f0 = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<10xf32>
  %1 = vector.transfer_read %0[%i], %f0 : memref<10xf32>, vector<4xf32>
  return %1: vector<4xf32>
}

// CHECK:         %[[SUBSPAN:.+]] = hal.interface.binding.subspan
// CHECK-COUNT-4: memref.load %[[SUBSPAN]]


// -----

// CHECK-LABEL: func.func @scalarize_indivisible_vector_transfer_write_op
func.func @scalarize_indivisible_vector_transfer_write_op(%value: vector<4xf32>, %i: index) {
  %f0 = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<10xf32>
  vector.transfer_write %value, %0[%i] : vector<4xf32>, memref<10xf32>
  return
}

// CHECK:         %[[SUBSPAN:.+]] = hal.interface.binding.subspan
// CHECK-COUNT-4: memref.store %{{.+}}, %[[SUBSPAN]]

// -----

// CHECK-LABEL: func.func @vectorize_alloc_with_mma_load_store
//  CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @vectorize_alloc_with_mma_load_store(%i0: index, %i1: index) {
  %alloc = memref.alloc() : memref<32x32xf16, 3>
  %0 = gpu.subgroup_mma_load_matrix %alloc[%i0, %i1] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "COp">
  gpu.subgroup_mma_store_matrix %0, %alloc[%i0, %i1] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
  return
}

// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x4xvector<4xf32>, 3>
// CHECK: %[[IDX:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%[[I1]]]
// CHECK: %[[LD:.+]] = gpu.subgroup_mma_load_matrix %[[ALLOC]][%[[I0]], %[[IDX]]] {leadDimension = 4 : index} : memref<32x4xvector<4xf32>, 3> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK: gpu.subgroup_mma_store_matrix %[[LD]], %[[ALLOC]][%[[I0]], %[[IDX]]] {leadDimension = 4 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x4xvector<4xf32>, 3>

// -----

// CHECK-LABEL: func.func @vectorize_alloc_with_mma_load_store
//  CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @vectorize_alloc_with_mma_load_store(%i0: index, %i1: index) {
  %alloc = memref.alloc() : memref<32x32xf16, 3>
  %0 = gpu.subgroup_mma_load_matrix %alloc[%i0, %i1] {leadDimension = 16 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "COp">
  gpu.subgroup_mma_store_matrix %0, %alloc[%i0, %i1] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
  return
}

//      CHECK: affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()
//      CHECK: gpu.subgroup_mma_load_matrix
// CHECK-SAME:   leadDimension = 2 : index
//      CHECK: gpu.subgroup_mma_store_matrix
// CHECK-SAME:   leadDimension = 2 : index

// -----

// CHECK-LABEL: func.func @vectorize_alloc_with_mma_load_store_unaligned_case
func.func @vectorize_alloc_with_mma_load_store_unaligned_case(%i0: index, %i1: index) {
  %alloc = memref.alloc() : memref<32x32xf16, 3>
  %0 = gpu.subgroup_mma_load_matrix %alloc[%i0, %i1] {leadDimension = 18 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "COp">
  gpu.subgroup_mma_store_matrix %0, %alloc[%i0, %i1] {leadDimension = 18 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
  return
}

//  CHECK-NOT: affine.apply
//      CHECK: gpu.subgroup_mma_load_matrix
// CHECK-SAME:   leadDimension = 18 : index
//      CHECK: gpu.subgroup_mma_store_matrix
// CHECK-SAME:   leadDimension = 18 : index

// -----

// CHECK-LABEL: func.func @scalarize_vector_load_op
//  CHECK-SAME: (%[[ARG0:.+]]: index)
func.func @scalarize_vector_load_op(%i: index) -> vector<4xi32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<10x10xi32>
  %1 = vector.load %0[%c0, %i] : memref<10x10xi32>, vector<4xi32>
  return %1: vector<4xi32>
}

// CHECK: %[[INIT:.+]] = arith.constant dense<0> : vector<4xi32>
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[SUBSPAN:.+]] = hal.interface.binding.subspan
// CHECK: %[[LD0:.+]] = memref.load %[[SUBSPAN]][%[[C0]], %[[ARG0]]] : memref<10x10xi32>
// CHECK: %[[INSERT0:.+]] = vector.insert %[[LD0]], %[[INIT]] [0] : i32 into vector<4xi32>
// CHECK: %[[IDX1:.+]] = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%[[ARG0]]]
// CHECK: %[[LD1:.+]] = memref.load %[[SUBSPAN]][%[[C0]], %[[IDX1]]] : memref<10x10xi32>
// CHECK: %[[INSERT1:.+]] = vector.insert %[[LD1]], %[[INSERT0]] [1] : i32 into vector<4xi32>
// CHECK: %[[IDX2:.+]] = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%[[ARG0]]]
// CHECK: %[[LD2:.+]] = memref.load %[[SUBSPAN]][%[[C0]], %[[IDX2]]] : memref<10x10xi32>
// CHECK: %[[INSERT2:.+]] = vector.insert %[[LD2]], %[[INSERT1]] [2] : i32 into vector<4xi32>
// CHECK: %[[IDX3:.+]] = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%[[ARG0]]]
// CHECK: %[[LD3:.+]] = memref.load %[[SUBSPAN]][%[[C0]], %[[IDX3]]] : memref<10x10xi32>
// CHECK: %[[INSERT3:.+]] = vector.insert %[[LD3]], %[[INSERT2]] [3] : i32 into vector<4xi32>
// CHECK: return %[[INSERT3]]

// -----

// Test that the memref is not vectorized if the element type is a complex type.

// CHECK-LABEL: func.func @complex_memref
func.func @complex_memref(%x: index, %y: index) -> complex<f32> {
  // CHECK: hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8x32xcomplex<f32>>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<8x32xcomplex<f32>>
  %1 = memref.load %0[%x, %y] : memref<8x32xcomplex<f32>>
  return %1: complex<f32>
}
