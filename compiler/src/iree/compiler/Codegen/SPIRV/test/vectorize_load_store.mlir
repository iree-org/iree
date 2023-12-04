// RUN: iree-opt --split-input-file --iree-spirv-vectorize-load-store --canonicalize -cse --mlir-print-local-scope %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-spirv-vectorize-load-store --cse --mlir-print-local-scope %s | FileCheck %s --check-prefix=BASE

func.func @alloc_transfer_read_write_vector4_vector8(%arg0: memref<4096x4096xf32>, %x: index, %y: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<128x32xf32, 3>
  %v = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<4xf32>
  vector.transfer_write %v, %0[%x, %y] : vector<4xf32>, memref<128x32xf32, 3>
  %mat = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<8xf32>
  vector.transfer_write %mat, %0[%x, %y] : vector<8xf32>, memref<128x32xf32, 3>
  memref.dealloc %0 : memref<128x32xf32, 3>
  return
}

// BASE-LABEL: func @alloc_transfer_read_write_vector4_vector8
//  BASE-SAME: (%[[ARG:.+]]: memref<4096x1024xvector<4xf32>>, %[[IDX0:.+]]: index, %[[IDX1:.+]]: index)

//   BASE-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   BASE-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   BASE-DAG:   %[[C4:.+]] = arith.constant 4 : index

//       BASE:   %[[ALLOC:.+]] = memref.alloc() : memref<128x8xvector<4xf32>, 3>

//       BASE:   %[[OFFSET0:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%[[IDX1]], %[[C4]]]
//       BASE:   %[[LOAD:.+]] = memref.load %[[ARG]][%[[IDX0]], %[[OFFSET0]]]
//       BASE:   memref.store %[[LOAD]], %[[ALLOC]][%[[IDX0]], %[[OFFSET0]]]

//       BASE:   %[[OFFSET1:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%0, %[[C0]]]
//       BASE:   %[[LOAD1:.+]] = memref.load %[[ARG]][%[[IDX0]], %[[OFFSET1]]]
//       BASE:   %[[OFFSET2:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%0, %[[C1]]]
//       BASE:   %[[LOAD2:.+]] = memref.load %[[ARG]][%[[IDX0]], %[[OFFSET2]]]
//       BASE:   %[[VEC:.+]] = vector.shuffle %[[LOAD1]], %[[LOAD2]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>

//       BASE:   %[[VEC0:.+]] = vector.extract_strided_slice %[[VEC]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
//       BASE:   memref.store %[[VEC0]], %[[ALLOC]][%[[IDX0]], %[[OFFSET1]]]
//       BASE:   %[[VEC1:.+]] = vector.extract_strided_slice %[[VEC]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
//       BASE:   memref.store %[[VEC1]], %[[ALLOC]][%[[IDX0]], %4]

// -----

// Test that the memref is not vectorized if used by scalar load or store.

// CHECK-LABEL: func.func @dont_vectorize_scalar_load
//  CHECK-SAME: %[[ARG0:.+]]: memref<4096x4096xf32>
func.func @dont_vectorize_scalar_load(%arg0: memref<4096x4096xf32>, %x: index, %y: index) -> f32 {
  %s = memref.load %arg0[%x, %y] : memref<4096x4096xf32>
  return %s : f32
}

// -----

// CHECK-LABEL: func.func @resource_copy()
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
func.func @resource_copy() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x4096xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf32>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<4xf32>, memref<4096x4096xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_with_offset()
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%{{.*}}) : memref<2048x4096x1024xvector<4xf32>, strided<[4194304, 1024, 1], offset: ?>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<2048x4096x1024xvector<4xf32>, strided<[4194304, 1024, 1], offset: ?>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
func.func @resource_copy_with_offset() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %offset = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<2048x4096x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf32>
  %v = vector.transfer_read %0[%c0, %c0, %c0], %cst : memref<2048x4096x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, vector<4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<4xf32>, memref<4096x4096xf32>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_f16
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
func.func @resource_copy_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x4096xf16>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<4xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<4xf16>, memref<4096x4096xf16>
  return
}

// -----

// CHECK-LABEL: func.func @resource_copy_8xf16
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x512xvector<4xf32>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x512xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
func.func @resource_copy_8xf16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4096x4096xf16>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<8xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<8xf16>, memref<4096x4096xf16>
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

  // CHECK: %[[VAL:.+]] = memref.load %[[INPUT]]
  // CHECK: memref.store %[[VAL]], %[[OUTPUT]]
  %v = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst : memref<?x8x?x128xf32>, vector<4xf32>
  vector.transfer_write %v, %1[%c0, %c0, %c0, %c0] : vector<4xf32>, memref<?x8x?x128xf32>

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
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x?xf32>, vector<4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<4xf32>, memref<4096x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @dont_vectorize_odd_vector_size
func.func @dont_vectorize_odd_vector_size() {
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
  // CHECK: %[[EXT_0:.+]] = vector.extract %{{.*}}[0] : f32 from vector<3xf32>
  // CHECK: memref.store %[[EXT_0]], %{{.*}}[%[[INDEX0]]] : memref<20xf32>
  // CHECK: %[[EXT_1:.+]] = vector.extract %{{.*}}[1] : f32 from vector<3xf32>
  // CHECK: memref.store %[[EXT_1]], %{{.*}}[%[[INDEX1]]] : memref<20xf32>
  // CHECK: %[[EXT_2:.+]] = vector.extract %{{.*}}[2] : f32 from vector<3xf32>
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
// CHECK: %[[E0:.+]] = vector.extract %[[VALUE]][0] : f32 from vector<4xf32>
// CHECK: memref.store %[[E0]], %[[BUFFER]][%[[C0]], %[[I1]], %[[I2]], %[[C0]]]
// CHECK: %[[PLUS1:.+]] = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%[[I2]]]
// CHECK: %[[E1:.+]] = vector.extract %[[VALUE]][1] : f32 from vector<4xf32>
// CHECK: memref.store %[[E1]], %[[BUFFER]][%[[C0]], %[[I1]], %[[PLUS1]], %[[C0]]]
// CHECK: %[[PLUS2:.+]] = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%[[I2]]]
// CHECK: %[[E2:.+]] = vector.extract %[[VALUE]][2] : f32 from vector<4xf32>
// CHECK: memref.store %[[E2]], %[[BUFFER]][%[[C0]], %[[I1]], %[[PLUS2]], %[[C0]]]
// CHECK: %[[PLUS3:.+]] = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%[[I2]]]
// CHECK: %[[E3:.+]] = vector.extract %[[VALUE]][3] : f32 from vector<4xf32>
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

// -----

// CHECK-LABEL: func.func @vectorize_mma_load_store_non_identity_memref
//  CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @vectorize_mma_load_store_non_identity_memref(%i0: index, %i1: index) {
  %c0 = arith.constant 0 : index
  %span0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1280xf16, strided<[1280, 1], offset: 11840>, #hal.descriptor_type<storage_buffer>>
  %span1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x1280xf16, strided<[1280, 1], offset: 11840>, #hal.descriptor_type<storage_buffer>>
  %val = gpu.subgroup_mma_load_matrix %span0[%i0, %i1] {leadDimension = 1280 : index} : memref<32x1280xf16, strided<[1280, 1], offset: 11840>, #hal.descriptor_type<storage_buffer>> -> !gpu.mma_matrix<16x16xf16, "COp">
  gpu.subgroup_mma_store_matrix %val, %span1[%i0, %i1] {leadDimension = 1280 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x1280xf16, strided<[1280, 1], offset: 11840>, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[SPAN0:.+]] = hal.interface.binding.subspan {{.+}} offset(%[[C0]]) flags(ReadOnly) : memref<32x160xvector<4xf32>, strided<[160, 1], offset: 1480>, #hal.descriptor_type<storage_buffer>>
// CHECK: %[[SPAN1:.+]] = hal.interface.binding.subspan {{.+}} offset(%[[C0]]) : memref<32x160xvector<4xf32>, strided<[160, 1], offset: 1480>, #hal.descriptor_type<storage_buffer>>
// CHECK: %[[APPLY:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%[[I1]]]
// CHECK: %[[VAL:.+]] = gpu.subgroup_mma_load_matrix %[[SPAN0]][%[[I0]], %[[APPLY]]] {leadDimension = 160 : index}
// CHECK: gpu.subgroup_mma_store_matrix %[[VAL]], %[[SPAN1]][%[[I0]], %[[APPLY]]] {leadDimension = 160 : index}

// -----

func.func @transfer_read_i4_memref_vector8(%x: index) -> vector<8xi4> {
  %c0_i4 = arith.constant 0 : i4
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048xi4>
  %1 = vector.transfer_read %0[%x], %c0_i4 {in_bounds = [true]} : memref<2048xi4>, vector<8xi4>
  return %1: vector<8xi4>
}

// CHECK-LABEL: func.func @transfer_read_i4_memref_vector8
//  CHECK-SAME: (%[[ARG:.+]]: index)
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<256xvector<1xi32>>
//       CHECK:   %[[INDEX:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%[[ARG]]]
//       CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[INDEX]]] : memref<256xvector<1xi32>>
//       CHECK:   %[[CAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi32> to vector<8xi4>
//       CHECK:   return %[[CAST]] : vector<8xi4>

// -----

// func.func @transfer_read_i4_memref_vector4(%x: index) -> vector<4xi4> {
//   %c0_i4 = arith.constant 0 : i4
//   %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048xi4>
//   %1 = vector.transfer_read %0[%x], %c0_i4 {in_bounds = [true]} : memref<2048xi4>, vector<4xi4>
//   return %1: vector<4xi4>
// }

// XXXXX-LABEL: func.func @transfer_read_i4_memref_vector4
//  XXXXX-SAME: (%[[ARG:.+]]: index)
//       XXXXX:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<512xvector<2xi8>>
//       XXXXX:   %[[INDEX:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 4)>()[%[[ARG]]]
//       XXXXX:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[INDEX]]] : memref<512xvector<2xi8>>
//       XXXXX:   %[[CAST:.+]] = vector.bitcast %[[LOAD]] : vector<2xi8> to vector<4xi4>
//       XXXXX:   return %[[CAST]] : vector<4xi4>

// -----

func.func @transfer_read_i4_memref_vector2(%x: index) -> vector<2xi4> {
  %c0_i4 = arith.constant 0 : i4
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048xi4>
  %1 = vector.transfer_read %0[%x], %c0_i4 {in_bounds = [true]} : memref<2048xi4>, vector<2xi4>
  return %1: vector<2xi4>
}

// XXXXX-LABEL: func.func @transfer_read_i4_memref_vector2
//  XXXXX-SAME: (%[[ARG:.+]]: index)
//       XXXXX:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1024xvector<1xi8>>
//       XXXXX:   %[[INDEX:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%[[ARG]]]
//       XXXXX:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[INDEX]]] : memref<1024xvector<1xi8>>
//       XXXXX:   %[[CAST:.+]] = vector.bitcast %[[LOAD]] : vector<1xi8> to vector<2xi4>
//       XXXXX:   return %[[CAST]] : vector<2xi4>

// -----

func.func @transfer_read_i3_memref_vector8(%x: index) -> vector<8xi3> {
  %c0_i3 = arith.constant 0 : i3
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048xi3>
  %1 = vector.transfer_read %0[%x], %c0_i3 {in_bounds = [true]} : memref<2048xi3>, vector<8xi3>
  return %1: vector<8xi3>
}

//   CHECK-LABEL: func.func @transfer_read_i3_memref_vector8
//         CHECK:   hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048xi3>
// CHECK-COUNT-8:   memref.load {{.+}} : memref<2048xi3>

// -----

func.func @transfer_read_vector2_vector8(%x: index) -> (vector<2xi32>, vector<8xi32>) {
  %c0 = arith.constant 0 : i32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048xi32>
  %1 = vector.transfer_read %0[%x], %c0 {in_bounds = [true]} : memref<2048xi32>, vector<2xi32>
  %2 = vector.transfer_read %0[%x], %c0 {in_bounds = [true]} : memref<2048xi32>, vector<8xi32>
  return %1, %2: vector<2xi32>, vector<8xi32>
}

// CHECK-LABEL: func @transfer_read_vector2_vector8
//  CHECK-SAME: (%[[INDEX:.+]]: index) -> (vector<2xi32>, vector<8xi32>)
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0> : vector<8xi32>
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan {{.+}} : memref<1024xvector<2xi32>>
//       CHECK:   %[[OFFSET0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%[[INDEX]]]
//       CHECK:   %[[LOAD0:.+]] = memref.load %[[SUBSPAN]][%[[OFFSET0]]]
//       CHECK:   %[[OFFSE1:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2 + 1)>()[%[[INDEX]]]
//       CHECK:   %[[LOAD1:.+]] = memref.load %[[SUBSPAN]][%[[OFFSE1]]]
//       CHECK:   %[[OFFSET2:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2 + 2)>()[%[[INDEX]]]
//       CHECK:   %[[LOAD2:.+]] = memref.load %[[SUBSPAN]][%[[OFFSET2]]]
//       CHECK:   %[[OFFSET3:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2 + 3)>()[%[[INDEX]]]
//       CHECK:   %[[LOAD3:.+]] = memref.load %[[SUBSPAN]][%[[OFFSET3]]]
//       CHECK:   %[[INSERT0:.+]] = vector.insert_strided_slice %[[LOAD0]], %[[INIT]] {offsets = [0], strides = [1]} : vector<2xi32> into vector<8xi32>
//       CHECK:   %[[INSERT1:.+]] = vector.insert_strided_slice %[[LOAD1]], %[[INSERT0]] {offsets = [2], strides = [1]} : vector<2xi32> into vector<8xi32>
//       CHECK:   %[[INSERT2:.+]] = vector.insert_strided_slice %[[LOAD2]], %[[INSERT1]] {offsets = [4], strides = [1]} : vector<2xi32> into vector<8xi32>
//       CHECK:   %[[INSERT3:.+]] = vector.insert_strided_slice %[[LOAD3]], %[[INSERT2]] {offsets = [6], strides = [1]} : vector<2xi32> into vector<8xi32>
//       CHECK:   return %[[LOAD0]], %[[INSERT3]]

// -----

func.func @transfer_write_vector2_vector8(%x: index, %val0: vector<2xi32>, %val1: vector<8xi32>) {
  %c0 = arith.constant 0 : i32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048xi32>
  vector.transfer_write %val0, %0[%x] : vector<2xi32>, memref<2048xi32>
  vector.transfer_write %val1, %0[%x] : vector<8xi32>, memref<2048xi32>
  return
}

// CHECK-LABEL: func @transfer_write_vector2_vector8
//  CHECK-SAME: (%[[INDEX:.+]]: index, %[[VAL0:.+]]: vector<2xi32>, %[[VAL1:.+]]: vector<8xi32>)
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1024xvector<2xi32>>

//       CHECK:   %[[OFFSET0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%[[INDEX]]]
//       CHECK:   memref.store %[[VAL0]], %[[SUBSPAN]][%[[OFFSET0]]]

//       CHECK:   %[[SLICE0:.+]] = vector.extract_strided_slice %[[VAL1]] {offsets = [0], sizes = [2], strides = [1]} : vector<8xi32> to vector<2xi32>
//       CHECK:   memref.store %[[SLICE0]], %[[SUBSPAN]][%[[OFFSET0]]]
//       CHECK:   %[[SLICE1:.+]] = vector.extract_strided_slice %[[VAL1]] {offsets = [2], sizes = [2], strides = [1]} : vector<8xi32> to vector<2xi32>
//       CHECK:   %[[OFFSET1:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2 + 1)>()[%[[INDEX]]]
//       CHECK:   memref.store %[[SLICE1]], %[[SUBSPAN]][%[[OFFSET1]]]
//       CHECK:   %[[SLICE2:.+]] = vector.extract_strided_slice %[[VAL1]] {offsets = [4], sizes = [2], strides = [1]} : vector<8xi32> to vector<2xi32>
//       CHECK:   %[[OFFSET2:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2 + 2)>()[%[[INDEX]]]
//       CHECK:   memref.store %[[SLICE2]], %[[SUBSPAN]][%[[OFFSET2]]]
//       CHECK:   %[[SLICE3:.+]] = vector.extract_strided_slice %[[VAL1]] {offsets = [6], sizes = [2], strides = [1]} : vector<8xi32> to vector<2xi32>
//       CHECK:   %[[OFFSET3:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2 + 3)>()[%[[INDEX]]]
//       CHECK:   memref.store %[[SLICE3]], %[[SUBSPAN]][%[[OFFSET3]]]

// -----

func.func @scalarize_masked_vector_transfer_op(%arg: vector<3xf32>, %mask: vector<3xi1>) -> (vector<3xf32>) {
  %c0 = arith.constant 0: index
  %c3 = arith.constant 3: index
  %f0 = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<20xf32>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<20xf32>
  %3 = vector.transfer_read %0[%c3], %f0, %mask : memref<20xf32>, vector<3xf32>
  vector.transfer_write %arg, %2[%c3], %mask : vector<3xf32>, memref<20xf32>
  return %3: vector<3xf32>
}

// CHECK-LABEL: func.func @scalarize_masked_vector_transfer_op
// CHECK-DAG: %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<3xf32>
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[PAD:.+]] = arith.constant 0.000000e+00 : f32

/// Transfer read.
//     CHECK: %[[MB0:.+]] = vector.extract %{{.*}}[0] : i1 from vector<3xi1>
//     CHECK: %[[MASK_LD0:.+]] = scf.if %[[MB0]] -> (f32) {
//     CHECK:   %[[LD0:.+]] = memref.load {{.*}}[%[[C3]]] : memref<20xf32>
//     CHECK:   scf.yield %[[LD0]] : f32
//     CHECK: } else {
//     CHECK:   scf.yield %[[PAD]] : f32
//     CHECK: }
//     CHECK: vector.insert %[[MASK_LD0]], %[[INIT]] [0] : f32 into vector<3xf32>
//     CHECK: vector.extract %{{.*}}[1] : i1 from vector<3xi1>
//     CHECK: scf.if %{{.*}} -> (f32) {
//     CHECK:   memref.load %{{.*}}[%[[C4]]] : memref<20xf32>
//     CHECK: vector.extract %{{.*}}[2] : i1 from vector<3xi1>
//     CHECK: scf.if %{{.*}} -> (f32) {
//     CHECK:   memref.load %{{.*}}[%[[C5]]] : memref<20xf32>
//     CHECK: %[[MASK_TR:.+]] = vector.insert {{.*}} [2] : f32 into vector<3xf32>

/// Transfer write.
//     CHECK: scf.if %[[MB0]] {
//     CHECK:   %[[E0:.+]] = vector.extract {{.*}}[0] : f32 from vector<3xf32>
//     CHECK:   memref.store %[[E0]], %{{.*}}[%[[C3]]] : memref<20xf32>
//     CHECK: }
//     CHECK: scf.if %{{.*}} {
//     CHECK:   %[[E1:.+]] = vector.extract {{.*}}[1] : f32 from vector<3xf32>
//     CHECK:   memref.store %[[E1]], %{{.*}}[%[[C4]]] : memref<20xf32>
//     CHECK: }
//     CHECK: scf.if %{{.*}} {
//     CHECK:   %[[E2:.+]] = vector.extract {{.*}}[2] : f32 from vector<3xf32>
//     CHECK:   memref.store %[[E2]], %{{.*}}[%[[C5]]] : memref<20xf32>
//     CHECK: }
//     CHECK: return %[[MASK_TR]] : vector<3xf32>

// -----

func.func @extract_vector_transfer_read_mask_bits(%arg: vector<3xf32>, %index: index) -> (vector<3xf32>) {
  %c3 = arith.constant 3: index
  %f0 = arith.constant 0.0 : f32
  %mask = vector.create_mask %index : vector<3xi1>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<20xf32>
  %1 = vector.transfer_read %0[%c3], %f0, %mask : memref<20xf32>, vector<3xf32>
  return %1: vector<3xf32>
}

// CHECK-LABEL: func.func @extract_vector_transfer_read_mask_bits
// CHECK-SAME:    %{{.*}}: vector<3xf32>, %[[MASK_SIZE:.+]]: index
// CHECK-DAG: %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<3xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[PAD:.+]] = arith.constant 0.000000e+00 : f32

//     CHECK: %[[MB0:.+]] = arith.cmpi sgt, %[[MASK_SIZE]], %[[C0]] : index
//     CHECK: %[[MASK_LD0:.+]] = scf.if %[[MB0]] -> (f32) {
//     CHECK:   %[[LD0:.+]] = memref.load {{.*}}[%[[C3]]] : memref<20xf32>
//     CHECK:   scf.yield %[[LD0]] : f32
//     CHECK: } else {
//     CHECK:   scf.yield %[[PAD]] : f32
//     CHECK: }
//     CHECK: vector.insert %[[MASK_LD0]], %[[INIT]] [0] : f32 into vector<3xf32>
//     CHECK: %[[MB1:.+]] = arith.cmpi sgt, %[[MASK_SIZE]], %[[C1]] : index
//     CHECK: scf.if %[[MB1]] -> (f32) {
//     CHECK:   memref.load %{{.*}}[%[[C4]]] : memref<20xf32>
//     CHECK: %[[MB2:.+]] = arith.cmpi sgt, %[[MASK_SIZE]], %[[C2]] : index
//     CHECK: scf.if %[[MB2]] -> (f32) {
//     CHECK:   memref.load %{{.*}}[%[[C5]]] : memref<20xf32>
