// RUN: iree-opt %s --split-input-file --iree-codegen-gpu-reduce-bank-conflicts  | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0 * 2048 + d1 * 64 + d2)>

// CHECK-LABEL: func.func @pad_alloc
func.func @pad_alloc(%a: memref<1024x1024xf32>) {
// CHECK: %[[A:.*]] = memref.alloc() : memref<4x32x68xf32, #gpu.address_space<workgroup>>
  %0 = memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
// CHECK: %[[S1:.*]] = memref.subview %[[A]][0, 0, 0] [4, 32, 64] [1, 1, 1] : memref<4x32x68xf32, #gpu.address_space<workgroup>> to memref<4x32x64xf32, strided<[2176, 68, 1]>, #gpu.address_space<workgroup>>
// CHECK: %[[S2:.*]] = memref.subview %[[S1]][0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, strided<[2176, 68, 1]>, #gpu.address_space<workgroup>> to memref<1x32x64xf32, strided<[2176, 68, 1]>, #gpu.address_space<workgroup>>
  %1 = memref.subview %0[0, 0, 0] [1, 32, 64] [1, 1, 1] :
    memref<4x32x64xf32, #gpu.address_space<workgroup>> to memref<1x32x64xf32, #map, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %2 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} :
    memref<1024x1024xf32>, vector<4xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[S2]][%{{.*}}, %{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<1x32x64xf32, strided<[2176, 68, 1]>, #gpu.address_space<workgroup>
  vector.transfer_write %2, %1[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<1x32x64xf32, #map, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @pad_alloc_negative
func.func @pad_alloc_negative(%a: memref<1024x1024xf32>, %i: index, %v: vector<4xf32>) {
  %c0 = arith.constant 0 : index
// CHECK: memref.alloc(%{{.*}}) : memref<?x32x64xf32, #gpu.address_space<workgroup>
  %0 = memref.alloc(%i) : memref<?x32x64xf32, #gpu.address_space<workgroup>>
  vector.transfer_write %v, %0[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<?x32x64xf32, #gpu.address_space<workgroup>>
  return
}
