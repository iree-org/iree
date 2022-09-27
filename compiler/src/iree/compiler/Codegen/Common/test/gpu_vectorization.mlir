// RUN: iree-opt --iree-codegen-gpu-vectorization %s | FileCheck %s

func.func @add_dispatch_0(%arg0: memref<1x8x4xf32>, %arg1: memref<1x4x8xf32>, %arg2: memref<1x4x8xf32>)  {
  linalg.generic {indexing_maps =
    [affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%arg0, %arg1 : memref<1x8x4xf32>, memref<1x4x8xf32>) outs(%arg2 : memref<1x4x8xf32>)
  attrs =  {__internal_linalg_transform__ = "vectorize"} {
  ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
    %19 = arith.addf %arg6, %arg7 : f32
    linalg.yield %19 : f32
  }
  return
}
// CHECK-LABEL: func.func @add_dispatch_0
// CHECK: vector.transfer_read {{.*}} : memref<1x8x4xf32>, vector<1x8x4xf32>
// CHECK: vector.transfer_read {{.*}} : memref<1x4x8xf32>, vector<1x4x8xf32>
// CHECK: addf %{{.*}}, %{{.*}} : vector<1x4x8xf32>
// CHECK: vector.transfer_write {{.*}} : vector<1x4x8xf32>, memref<1x4x8xf32>
