// RUN: iree-opt -split-input-file -iree-spirv-vectorize-memref -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: func @copy
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x1024xvector<4xf32>>
//       CHECK: %[[ALLOC:.+]] = alloc() : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[V:.+]] = load %[[ARG0]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//       CHECK: store %[[V]], %[[ALLOC]][%{{.*}}, %{{.*}}] : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[MAT:.+]] = vector.transfer_read %[[ARG0]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//       CHECK: vector.transfer_write %[[MAT]], %[[ALLOC]][%{{.*}}, %{{.*}}] : vector<32x8xf32>, memref<128x8xvector<4xf32>, 3>
//       CHECK: dealloc %[[ALLOC]] : memref<128x8xvector<4xf32>, 3>
func @copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<128x32xf32, 3>
  %v = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<1x4xf32>
  vector.transfer_write %v, %0[%x, %y] : vector<1x4xf32>, memref<128x32xf32, 3>
  %mat = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %0[%x, %y] : vector<32x8xf32>, memref<128x32xf32, 3>
  dealloc %0 : memref<128x32xf32, 3>
  return
}

// -----

// Test that the memref is not vectorized if used by scalar load or store.
// CHECK-LABEL: func @copy
//  CHECK-SAME: %[[ARG0:.+]]: memref<4096x4096xf32>
func @copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<128x32xf32, 3>
  %s = load %arg0[%x, %y] : memref<4096x4096xf32>
  store %s, %0[%x, %y] : memref<128x32xf32, 3>
  dealloc %0 : memref<128x32xf32, 3>
  return
}

// -----

// CHECK-LABEL: func @resource_copy
//     CHECK: %[[A:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[B:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[V:.+]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf32>, memref<4096x1024xvector<4xf32>>
func @resource_copy() {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x4096xf32>
  %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x4096xf32>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<1x4xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf32>, memref<4096x4096xf32>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
  return
}

hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @resource_copy_f16
//     CHECK: %[[A:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[B:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[V:.+]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
//     CHECK: store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf16>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf16>>, vector<32x8xf16>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf16>, memref<4096x1024xvector<4xf16>>
func @resource_copy_f16() {
  %cst = constant 0.000000e+00 : f16
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x4096xf16>
  %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<1x4xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf16>, memref<4096x4096xf16>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<32x8xf16>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf16>, memref<4096x4096xf16>
  return
}

hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @resource_copy_8xf16
//     CHECK: %[[A:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x512xvector<4xf32>>
//     CHECK: %[[B:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x512xvector<4xf32>>
//     CHECK: %[[V:.+]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
//     CHECK: store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x512xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x512xvector<4xf32>>, vector<32x8xf16>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf16>, memref<4096x512xvector<4xf32>>
func @resource_copy_8xf16() {
  %cst = constant 0.000000e+00 : f16
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x4096xf16>
  %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x4096xf16>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<1x8xf16>
  vector.transfer_write %v, %1[%c0, %c0] : vector<1x8xf16>, memref<4096x4096xf16>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf16>, vector<32x8xf16>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf16>, memref<4096x4096xf16>
  return
}

hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: func @do_not_vectorize_odd_vector_size
func @do_not_vectorize_odd_vector_size() {
  %cst = constant 0.0 : f32
  %c0 = constant 0 : index
  // CHECK: iree.placeholder
  // CHECK-SAME: memref<4x3xf32>
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4x3xf32>
  // CHECK: iree.placeholder
  // CHECK-SAME: memref<4x3xf32>
  %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4x3xf32>
  %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4x3xf32>, vector<3xf32>
  vector.transfer_write %v, %1[%c0, %c0] : vector<3xf32>, memref<4x3xf32>
  return
}

hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
}

// -----

func @vectorize_binding_subspan() {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  // CHECK: hal.interface.binding.subspan @legacy_io::@arg0[%c0]
  // CHECK-SAME: memref<4096x1024xvector<4xf32>>
  // CHECK: hal.interface.binding.subspan @legacy_io::@ret0[%c0]
  // CHECK-SAME: memref<4096x1024xvector<4xf32>>
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<4096x4096xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<4096x4096xf32>
  %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<32x8xf32>
  vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
  return
}

hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
