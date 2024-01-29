// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-to-gpu),canonicalize)" --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @copies_to_asyncs
func.func @copies_to_asyncs(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  // CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4
  %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
  // CHECK-NOT: nvgpu.device_async_create_group

  // CHECK: %[[CP1:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 1
  %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
  vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
  // CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]], %[[CP1]]
  // CHECK: nvgpu.device_async_wait %[[G]]
  return
}

// -----

func.func @ksplitmatmul_basic(%a: memref<128x16x256xf32>) -> vector<16x1x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4], %cst {in_bounds = [true, true, true]} : memref<128x16x256xf32>, vector<16x1x8xf32>
  return %0 : vector<16x1x8xf32>
}
// CHECK-LABEL: func.func @ksplitmatmul_basic
//   CHECK-DAG: %[[ID:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: %[[M:.*]] = memref.subview
//  CHECK-SAME:[2, 3, 4] [16, 1, 8] [1, 1, 1]
//  CHECK-SAME:memref<128x16x256xf32> to memref<16x8xf32, strided<[4096, 1], offset: 8964>>
//       CHECK: vector.transfer_read %[[M]][%[[ID]], %[[ID]]]
//  CHECK-SAME: {in_bounds = [true, true]} : memref<16x8xf32, strided<[4096, 1], offset: 8964>>, vector<16x8xf32>
//       CHECK: vector.broadcast %{{.*}} : vector<16x8xf32> to vector<1x16x8xf32>
//       CHECK: vector.transpose %{{.*}} [1, 0, 2] : vector<1x16x8xf32> to vector<16x1x8xf32>
//       CHECK: return %{{.*}} : vector<16x1x8xf32>

// -----

func.func @ksplitmatmul_nounitdim(%a: memref<128x16x256xf32>) -> vector<16x2x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4], %cst {in_bounds = [true, true, true]} : memref<128x16x256xf32>, vector<16x2x8xf32>
  return %0 : vector<16x2x8xf32>
}
// CHECK-LABEL: func.func @ksplitmatmul_nounitdim
//   CHECK-DAG: %[[ID:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[ID2:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[ID3:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: vector.transfer_read %{{.*}}[%[[ID]], %[[ID2]], %[[ID3]]]
//  CHECK-SAME: {in_bounds = [true, true, true]} : memref<128x16x256xf32>, vector<16x2x8xf32>
//       CHECK: return %{{.*}} : vector<16x2x8xf32>

// -----

func.func @ksplitmatmul_4D(%a: memref<128x16x32x256xf32>) -> vector<16x1x1x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4, %c5], %cst {in_bounds = [true, true, true, true]} : memref<128x16x32x256xf32>, vector<16x1x1x8xf32>
  return %0 : vector<16x1x1x8xf32>
}
// CHECK-LABEL: func.func @ksplitmatmul_4D
//   CHECK-DAG: %[[ID:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: %[[M:.*]] = memref.subview
//  CHECK-SAME:[2, 3, 4, 5] [16, 1, 1, 8] [1, 1, 1, 1]
//  CHECK-SAME: memref<128x16x32x256xf32> to memref<16x8xf32, strided<[131072, 1], offset: 287749>>
//       CHECK: vector.transfer_read %[[M]][%[[ID]], %[[ID]]]
//  CHECK-SAME: {in_bounds = [true, true]} :  memref<16x8xf32, strided<[131072, 1], offset: 287749>>, vector<16x8xf32>
//       CHECK: vector.broadcast %{{.*}} : vector<16x8xf32> to vector<1x1x16x8xf32>
//       CHECK: vector.transpose %{{.*}} [2, 0, 1, 3] : vector<1x1x16x8xf32> to vector<16x1x1x8xf32>
//       CHECK: return %{{.*}} : vector<16x1x1x8xf32>

// -----

func.func @ksplitmatmul_4D_lower_rank_read(%a: memref<128x512x32x256xf32>) -> vector<16x1x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4, %c5], %cst {in_bounds = [true, true, true]} : memref<128x512x32x256xf32>, vector<16x1x8xf32>
  return %0 : vector<16x1x8xf32>
}
// CHECK-LABEL: func.func @ksplitmatmul_4D_lower_rank_read
//   CHECK-DAG: %[[ID:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: %[[M:.*]] = memref.subview
//  CHECK-SAME:[2, 3, 4, 5] [1, 16, 1, 8] [1, 1, 1, 1]
//  CHECK-SAME: memref<128x512x32x256xf32> to memref<16x8xf32, strided<[8192, 1], offset: 8414213>>
//       CHECK: vector.transfer_read %[[M]][%[[ID]], %[[ID]]]
//  CHECK-SAME: {in_bounds = [true, true]} :  memref<16x8xf32, strided<[8192, 1], offset: 8414213>>, vector<16x8xf32>
//       CHECK: vector.broadcast %{{.*}} : vector<16x8xf32> to vector<1x16x8xf32>
//       CHECK: vector.transpose %{{.*}} [1, 0, 2] : vector<1x16x8xf32> to vector<16x1x8xf32>
//       CHECK: return %{{.*}} : vector<16x1x8xf32>

// -----

func.func @ksplitmatmul_4D_negative(%a: memref<128x16x32x256xf32>) -> vector<16x1x8x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4, %c5], %cst {in_bounds = [true, true, true, true]} : memref<128x16x32x256xf32>, vector<16x1x8x1xf32>
  return %0 : vector<16x1x8x1xf32>
}

// CHECK-LABEL: func.func @ksplitmatmul_4D_negative
//   CHECK-DAG: %[[ID:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[ID2:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[ID3:.*]] = arith.constant 4 : index
//   CHECK-DAG: %[[ID4:.*]] = arith.constant 5 : index
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: vector.transfer_read %{{.*}}[%[[ID]], %[[ID2]], %[[ID3]], %[[ID4]]]
//  CHECK-SAME: {in_bounds = [true, true, true, true]} : memref<128x16x32x256xf32>, vector<16x1x8x1xf32>
//       CHECK: return %{{.*}} : vector<16x1x8x1xf32>

// -----

func.func @ksplitmatmul_4D_allone(%a: memref<128x16x32x256xf32>) -> vector<1x1x1x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4, %c5], %cst {in_bounds = [true, true, true, true]} : memref<128x16x32x256xf32>, vector<1x1x1x1xf32>
  return %0 : vector<1x1x1x1xf32>
}

// CHECK-LABEL: func.func @ksplitmatmul_4D_allone
//   CHECK-DAG: %[[ID:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: %[[M:.*]] = memref.subview
//  CHECK-SAME:[2, 3, 4, 5] [1, 1, 1, 1] [1, 1, 1, 1]
//  CHECK-SAME: memref<128x16x32x256xf32> to memref<1x1xf32, strided<[131072, 8192], offset: 287749>>
//       CHECK: vector.transfer_read %[[M]][%[[ID]], %[[ID]]]
//  CHECK-SAME: {in_bounds = [true]} :  memref<1x1xf32, strided<[131072, 8192], offset: 287749>>, vector<1xf32>
//       CHECK: vector.broadcast %{{.*}} : vector<1xf32> to vector<1x1x1x1xf32>
//   CHECK-NOT: vector.transpose
//       CHECK: return %{{.*}} : vector<1x1x1x1xf32>

// -----

// CHECK-LABEL: func.func @copies_to_asyncs_mask
//  CHECK-SAME: , %[[I:.+]]: index)
func.func @copies_to_asyncs_mask(%a: memref<1024x1024xf32>, %i: index) {
  %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %i : vector<4xi1>
  // CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4, %[[I]]
  %1 = vector.transfer_read %a[%c0, %c0], %cst_0, %mask {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
  // CHECK-NOT: nvgpu.device_async_create_group

  %mask1 = vector.create_mask %i : vector<1xi1>
  // CHECK: %[[CP1:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 1, %[[I]]
  %2 = vector.transfer_read %a[%c0, %c4], %cst_0, %mask1 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
  vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
  // CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]], %[[CP1]]
  // CHECK: nvgpu.device_async_wait %[[G]]

  %3 = vector.transfer_read %a[%c0, %c4], %cst_0, %mask1 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
  vector.transfer_write %3, %0[%c0, %c4, %c0], %mask1 {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
  // We cannot generate async copy if the write is masked.
  // CHECK-NOT: nvgpu.device_async_copy
  return
}

// -----

// CHECK-LABEL: func.func @copies_to_asyncs_mask_2d
//  CHECK-SAME: , %[[I:[0-9a-zA-Z]+]]: index
//  CHECK-SAME: , %[[J:[0-9a-zA-Z]+]]: index)
func.func @copies_to_asyncs_mask_2d(%a: memref<1024x1024xf32>, %i: index, %j: index) {
  %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %i, %j : vector<4x4xi1>

  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
  //     CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[I]], %[[c2]] : index
  //     CHECK: %[[CNT:.*]] = arith.select %[[CMP]], %[[J]], %[[c0]] : index
  //     CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4, %[[CNT]]
  %submask = vector.extract %mask[2] : vector<4xi1> from vector<4x4xi1>
  %1 = vector.transfer_read %a[%c0, %c0], %cst_0, %submask {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
  //     CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]]
  //     CHECK: nvgpu.device_async_wait %[[G]]

  return
}
