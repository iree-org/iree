// RUN: iree-opt -split-input-file -iree-spirv-vectorize-load-store -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: func @copy
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x1024xvector<4xf32>>
//       CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[V:.+]] = memref.load %[[ARG0]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//       CHECK: memref.store %[[V]], %[[ALLOC]][%{{.*}}, %{{.*}}] : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[MAT:.+]] = vector.transfer_read %[[ARG0]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//       CHECK: vector.transfer_write %[[MAT]], %[[ALLOC]][%{{.*}}, %{{.*}}] : vector<32x8xf32>, memref<128x8xvector<4xf32>, 3>
//       CHECK: memref.dealloc %[[ALLOC]] : memref<128x8xvector<4xf32>, 3>
func @copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) {
  %cst = constant 0.000000e+00 : f32
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
// CHECK-LABEL: func @copy
//  CHECK-SAME: %[[ARG0:.+]]: memref<4096x4096xf32>
func @copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) {
  %cst = constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<128x32xf32, 3>
  %s = memref.load %arg0[%x, %y] : memref<4096x4096xf32>
  memref.store %s, %0[%x, %y] : memref<128x32xf32, 3>
  memref.dealloc %0 : memref<128x32xf32, 3>
  return
}

// -----

// CHECK-LABEL: func @resource_copy
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf32>, memref<4096x1024xvector<4xf32>>
func @resource_copy() {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x4096xf32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x4096xf32>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<1x4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf32>, memref<4096x4096xf32>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
  return
}

hal.interface private @io attributes {push_constants = 5 : index} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @resource_copy_f16
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf16>>, vector<32x8xf16>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf16>, memref<4096x1024xvector<4xf16>>
func @resource_copy_f16() {
  %cst = constant 0.000000e+00 : f16
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x4096xf16>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<1x4xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf16>, memref<4096x4096xf16>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<32x8xf16>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf16>, memref<4096x4096xf16>
  return
}

hal.interface private @io attributes {push_constants = 5 : index} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @resource_copy_8xf16
//     CHECK: %[[A:.+]] = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x512xvector<4xf32>>
//     CHECK: %[[B:.+]] = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x512xvector<4xf32>>
//     CHECK: %[[V:.+]] = memref.load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
//     CHECK: memref.store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x512xvector<4xf32>>, vector<32x8xf16>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf16>, memref<4096x512xvector<4xf32>>
func @resource_copy_8xf16() {
  %cst = constant 0.000000e+00 : f16
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x4096xf16>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<1x8xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x8xf16>, memref<4096x4096xf16>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<32x8xf16>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf16>, memref<4096x4096xf16>
  return
}

hal.interface private @io attributes {push_constants = 5 : index} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @resource_copy_dynamic_shape()
func @resource_copy_dynamic_shape() {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  // CHECK: %[[DIM0:.+]] = hal.interface.load.constant offset = 0 : index
  // CHECK: %[[DIM1:.+]] = hal.interface.load.constant offset = 1 : index
  %dim0 = hal.interface.load.constant offset = 0 : index
  %dim1 = hal.interface.load.constant offset = 1 : index

  // CHECK: %[[INPUT:.+]] = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x8x?x32xvector<4xf32>>{%[[DIM0]], %[[DIM1]]}
  // CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x8x?x32xvector<4xf32>>{%[[DIM0]], %[[DIM1]]}
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x8x?x128xf32>{%dim0, %dim1}
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x8x?x128xf32>{%dim0, %dim1}

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

hal.interface @io attributes {push_constants = 5 : index, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @resource_copy_dynamic_last_dim()
func @resource_copy_dynamic_last_dim() {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %dim = hal.interface.load.constant offset = 0 : index
  // CHECK: hal.interface.binding.subspan @io::@arg0[{{.+}}] : memref<4096x?xf32>
  // CHECK: hal.interface.binding.subspan @io::@ret0[{{.+}}] : memref<4096x?xf32>
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x?xf32>{%dim}
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x?xf32>{%dim}
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x?xf32>, vector<1x4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf32>, memref<4096x?xf32>
  return
}

hal.interface @io attributes {push_constants = 5 : index, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @do_not_vectorize_odd_vector_size
func @do_not_vectorize_odd_vector_size() {
  %cst = constant 0.0 : f32
  %c0 = constant 0 : index
  // CHECK: hal.interface.binding.subspan
  // CHECK-SAME: memref<4x3xf32>
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x3xf32>
  // CHECK: hal.interface.binding.subspan
  // CHECK-SAME: memref<4x3xf32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x3xf32>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4x3xf32>, vector<3xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<3xf32>, memref<4x3xf32>
  return
}

hal.interface private @io  {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

func @vectorize_binding_subspan() {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  // CHECK: hal.interface.binding.subspan @io::@arg0[%c0]
  // CHECK-SAME: memref<4096x1024xvector<4xf32>>
  // CHECK: hal.interface.binding.subspan @io::@ret0[%c0]
  // CHECK-SAME: memref<4096x1024xvector<4xf32>>
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4096x4096xf32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4096x4096xf32>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
  return
}

hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}

// -----

// CHECK-LABEL: func @scalarize_vector_transfer_op
func @scalarize_vector_transfer_op(%arg: vector<3xf32>) -> (vector<3xf32>) {
  %c0 = constant 0: index
  %c3 = constant 3: index
  %f0 = constant 0.0 : f32
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<20xf32>
  %2 = hal.interface.binding.subspan @io::@ret1[%c0] : memref<20xf32>
  // CHECK-DAG: %[[INDEX0:.+]] = constant 3 : index
  // CHECK-DAG: %[[INDEX1:.+]] = constant 4 : index
  // CHECK-DAG: %[[INDEX2:.+]] = constant 5 : index
  // CHECK-DAG: %[[CST:.+]] = constant dense<0.000000e+00> : vector<3xf32>

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
