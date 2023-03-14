// RUN: iree-opt --iree-llvmgpu-tensorcore-vectorization %s | FileCheck %s

func.func @dot() {
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2048x1024xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<1024x512xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<2048x512xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %5 = memref.subview %0[%3, 0] [64, 1024] [1, 1] : memref<2048x1024xf32> to memref<64x1024xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
  %6 = memref.subview %1[0, %4] [1024, 64] [1, 1] : memref<1024x512xf32> to memref<1024x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>
  %7 = memref.subview %2[%3, %4] [64, 64] [1, 1] : memref<2048x512xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>
  %8 = gpu.thread_id x
  %9 = gpu.thread_id y
  %10 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%9]
  %11 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 32)>(%8)
  %12 = memref.subview %7[%10, %11] [32, 32] [1, 1] : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>> to memref<32x32xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>
  linalg.fill {__internal_linalg_transform__ = "vectorize"} ins(%cst : f32) outs(%12 : memref<32x32xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>)
  scf.for %arg0 = %c0 to %c1024 step %c16 {
    %13 = memref.subview %5[0, %arg0] [64, 16] [1, 1] : memref<64x1024xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
    %14 = memref.subview %6[%arg0, 0] [16, 64] [1, 1] : memref<1024x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>> to memref<16x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>
    %15 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 32)>(%8)
    %16 = memref.subview %13[%10, 0] [32, 16] [1, 1] : memref<64x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<32x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
    %17 = memref.subview %14[0, %15] [16, 32] [1, 1] : memref<16x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>> to memref<16x32xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>
    %18 = memref.subview %7[%10, %15] [32, 32] [1, 1] : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>> to memref<32x32xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>
    linalg.matmul {__internal_linalg_transform__ = "vectorize"} ins(%16, %17 : memref<32x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, memref<16x32xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>) outs(%18 : memref<32x32xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>>)
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
