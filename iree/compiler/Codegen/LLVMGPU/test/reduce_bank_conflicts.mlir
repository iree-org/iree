// RUN: iree-opt %s -iree-llvmgpu-reduce-bank-conflicts  | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0 * 2048 + d1 * 64 + d2)>
// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1, d2) -> (d0 * 2176 + d1 * 68 + d2)>

// CHECK-LABEL: func @pad_alloc
func.func @pad_alloc(%a: memref<1024x1024xf32>) {
// CHECK: %[[A:.*]] = memref.alloc() : memref<4x32x68xf32, 3>
  %0 = memref.alloc() : memref<4x32x64xf32, 3>
// CHECK: %[[S1:.*]] = memref.subview %[[A]][0, 0, 0] [4, 32, 64] [1, 1, 1] : memref<4x32x68xf32, 3> to memref<4x32x64xf32, #[[$MAP]], 3>
// CHECK: %[[S2:.*]] = memref.subview %[[S1]][0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<4x32x64xf32, #[[$MAP]], 3> to memref<1x32x64xf32, #[[$MAP]], 3>
  %1 = memref.subview %0[0, 0, 0] [1, 32, 64] [1, 1, 1] :
    memref<4x32x64xf32, 3> to memref<1x32x64xf32, #map, 3>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %2 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} :
    memref<1024x1024xf32>, vector<4xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[S2]][%{{.*}}, %{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<1x32x64xf32, #[[$MAP]], 3>    
  vector.transfer_write %2, %1[%c0, %c0, %c0] {in_bounds = [true]} : 
    vector<4xf32>, memref<1x32x64xf32, #map, 3>
  return
}
