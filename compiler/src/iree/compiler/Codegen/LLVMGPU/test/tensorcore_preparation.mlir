// RUN: iree-opt --iree-llvmgpu-tensorcore-preparation %s | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>
#map3 = affine_map<()[s0] -> (s0 * 32)>
#map4 = affine_map<(d0) -> ((d0 floordiv 32) * 32)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @dot() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048x1024xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<1024x512xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<2048x512xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply #map()[%workgroup_id_y]
  %4 = affine.apply #map()[%workgroup_id_x]
  %subview = memref.subview %0[%3, 0] [64, 1024] [1, 1] : memref<2048x1024xf32> to memref<64x1024xf32, #map1>
  %subview_1 = memref.subview %1[0, %4] [1024, 64] [1, 1] : memref<1024x512xf32> to memref<1024x64xf32, #map2>
  %subview_2 = memref.subview %2[%3, %4] [64, 64] [1, 1] : memref<2048x512xf32> to memref<64x64xf32, #map2>
  %5 = gpu.thread_id  x
  %6 = gpu.thread_id  y
  %7 = affine.apply #map3()[%6]
  %8 = affine.apply #map4(%5)
  %subview_3 = memref.subview %subview_2[%7, %8] [32, 32] [1, 1] : memref<64x64xf32, #map2> to memref<32x32xf32, #map2>
  vector.transfer_write %cst, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, #map2>
  scf.for %arg0 = %c0 to %c1024 step %c16 {
    %subview_4 = memref.subview %subview[0, %arg0] [64, 16] [1, 1] : memref<64x1024xf32, #map1> to memref<64x16xf32, #map1>
    %subview_5 = memref.subview %subview_1[%arg0, 0] [16, 64] [1, 1] : memref<1024x64xf32, #map2> to memref<16x64xf32, #map2>
    %9 = affine.apply #map4(%5)
    %subview_6 = memref.subview %subview_4[%7, 0] [32, 16] [1, 1] : memref<64x16xf32, #map1> to memref<32x16xf32, #map1>
    %subview_7 = memref.subview %subview_5[0, %9] [16, 32] [1, 1] : memref<16x64xf32, #map2> to memref<16x32xf32, #map2>
    %subview_8 = memref.subview %subview_2[%7, %9] [32, 32] [1, 1] : memref<64x64xf32, #map2> to memref<32x32xf32, #map2>
    %10 = vector.transfer_read %subview_6[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x16xf32, #map1>, vector<32x16xf32>
    %11 = vector.transfer_read %subview_7[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x32xf32, #map2>, vector<16x32xf32>
    %12 = vector.transfer_read %subview_8[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32, #map2>, vector<32x32xf32>
    %13 = vector.contract {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %10, %11, %12 : vector<32x16xf32>, vector<16x32xf32> into vector<32x32xf32>
    vector.transfer_write %13, %subview_8[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, #map2>
  }
  return
}
//    CHECK-LABEL: func.func @dot
// CHECK-COUNT-4:   vector.transfer_write {{.*}} : vector<16x16xf32>, memref<32x32xf32
//         CHECK:   scf.for
// CHECK-COUNT-4:     vector.transfer_read {{.*}} {in_bounds = [true, true]} : memref<32x16xf32, #{{.*}}>, vector<16x8xf32>
// CHECK-COUNT-4:     vector.transfer_read {{.*}} {in_bounds = [true, true]} : memref<16x32xf32, #{{.*}}>, vector<8x16xf32>
// CHECK-COUNT-4:     vector.transfer_read {{.*}} {in_bounds = [true, true]} : memref<32x32xf32, #{{.*}}>, vector<16x16xf32>
// CHECK-COUNT-8:     vector.contract {{.*}} : vector<16x8xf32>, vector<8x16xf32> into vector<16x16xf32>
// CHECK-COUNT-4:     vector.transfer_write {{.*}} : vector<16x16xf32>, memref<32x32xf32
//    CHECK-NEXT:   }
