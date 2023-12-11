// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmcpu-vector-lowering{split-transfers=linalg-copy}))' --split-input-file %s | FileCheck %s

func.func @matmul_391x384x384_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c384 = arith.constant 384 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %cst_1 = arith.constant dense<6.000000e+00> : vector<8x32xf32>
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<8x32xf32>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<391x384xf32>
  memref.assume_alignment %0, 64 : memref<391x384xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<384x384xf32>
  memref.assume_alignment %1, 64 : memref<384x384xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<384xf32>
  memref.assume_alignment %2, 64 : memref<384xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<391x384xf32>
  memref.assume_alignment %3, 64 : memref<391x384xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %6 = affine.min affine_map<(d0) -> (-d0 + 391, 128)>(%4)
  %subview = memref.subview %0[%4, 0] [%6, 384] [1, 1] : memref<391x384xf32> to memref<?x384xf32, strided<[384, 1], offset: ?>>
  %subview_2 = memref.subview %3[%4, %5] [%6, 128] [1, 1] : memref<391x384xf32> to memref<?x128xf32, strided<[384, 1], offset: ?>>
  %subview_3 = memref.subview %2[%5] [128] [1] : memref<384xf32> to memref<128xf32, strided<[1], offset: ?>>
  %subview_4 = memref.subview %1[0, %5] [384, 128] [1, 1] : memref<384x384xf32> to memref<384x128xf32, strided<[384, 1], offset: ?>>
  scf.for %arg0 = %c0 to %6 step %c8 {
    %7 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg0)[%6]
    %subview_5 = memref.subview %subview[%arg0, 0] [%7, 384] [1, 1] : memref<?x384xf32, strided<[384, 1], offset: ?>> to memref<?x384xf32, strided<[384, 1], offset: ?>>
    scf.for %arg1 = %c0 to %c128 step %c32 {
      vector.transfer_write %cst_0, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32>
      %subview_6 = memref.subview %alloca[0, 0] [%7, 32] [1, 1] : memref<8x32xf32> to memref<?x32xf32, strided<[32, 1]>>
      %8 = vector.transfer_read %subview_6[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x32xf32, strided<[32, 1]>>, vector<8x32xf32>
      %9 = scf.for %arg2 = %c0 to %c384 step %c16 iter_args(%arg3 = %8) -> (vector<8x32xf32>) {
        %15 = vector.transfer_read %subview_5[%c0, %arg2], %cst : memref<?x384xf32, strided<[384, 1], offset: ?>>, vector<8x16xf32>
        %16 = vector.transfer_read %subview_4[%arg2, %arg1], %cst {in_bounds = [true, true]} : memref<384x128xf32, strided<[384, 1], offset: ?>>, vector<16x32xf32>
        %17 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %16, %arg3 : vector<8x16xf32>, vector<16x32xf32> into vector<8x32xf32>
        scf.yield %17 : vector<8x32xf32>
      }
      %10 = vector.transfer_read %subview_3[%arg1], %cst {in_bounds = [true]} : memref<128xf32, strided<[1], offset: ?>>, vector<32xf32>
      %11 = vector.broadcast %10 : vector<32xf32> to vector<8x32xf32>
      %12 = arith.addf %11, %9 : vector<8x32xf32>
      %13 = arith.minimumf %12, %cst_1 : vector<8x32xf32>
      %14 = arith.maximumf %13, %cst_0 : vector<8x32xf32>
      %subview_7 = memref.subview %subview_2[%arg0, %arg1] [%7, 32] [1, 1] : memref<?x128xf32, strided<[384, 1], offset: ?>> to memref<?x32xf32, strided<[384, 1], offset: ?>>
      vector.transfer_write %14, %subview_7[%c0, %c0] {in_bounds = [false, true]} : vector<8x32xf32>, memref<?x32xf32, strided<[384, 1], offset: ?>>
    }
  }
  return
}

// CHECK-LABEL:       func.func @matmul_391x384x384_f32
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   %{{.*}} scf.if
// CHECK:                     scf.yield %{{.*}}
// CHECK:                   } else {
// CHECK:                     linalg.fill
// CHECK:                     memref.copy
// CHECK:                     scf.yield %{{.*}}
// CHECK:                   }
// CHECK:                   scf.for
// CHECK-COUNT-128:           vector.fma
// CHECK:                 %{{.*}} = scf.if
// CHECK:                 scf.if %{{.*}} {
// CHECK:                   memref.copy
// CHECK:                 }

// -----

// Check that vector.loads whose elements are extracted and
// consumed in a scalar fashion are scalarized.

func.func @matmul_scalar_loads() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c384 = arith.constant 384 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<8x32xf32>
  %cst_1 = arith.constant dense<6.000000e+00> : vector<8x32xf32>
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<8x32xf32>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<391x384xf32>
  memref.assume_alignment %0, 64 : memref<391x384xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<384x384xf32>
  memref.assume_alignment %1, 64 : memref<384x384xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<384xf32>
  memref.assume_alignment %2, 64 : memref<384xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<391x384xf32>
  memref.assume_alignment %3, 64 : memref<391x384xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %6 = affine.min affine_map<(d0) -> (-d0 + 391, 128)>(%4)
  %subview = memref.subview %0[%4, 0] [%6, 384] [1, 1] : memref<391x384xf32> to memref<?x384xf32, strided<[384, 1], offset: ?>>
  %subview_2 = memref.subview %3[%4, %5] [%6, 128] [1, 1] : memref<391x384xf32> to memref<?x128xf32, strided<[384, 1], offset: ?>>
  %subview_3 = memref.subview %2[%5] [128] [1] : memref<384xf32> to memref<128xf32, strided<[1], offset: ?>>
  %subview_4 = memref.subview %1[0, %5] [384, 128] [1, 1] : memref<384x384xf32> to memref<384x128xf32, strided<[384, 1], offset: ?>>
  scf.for %arg0 = %c0 to %6 step %c8 {
    %7 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg0)[%6]
    %subview_5 = memref.subview %subview[%arg0, 0] [%7, 384] [1, 1] : memref<?x384xf32, strided<[384, 1], offset: ?>> to memref<?x384xf32, strided<[384, 1], offset: ?>>
    scf.for %arg1 = %c0 to %c128 step %c32 {
      vector.transfer_write %cst_0, %alloca[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32>
      %subview_6 = memref.subview %alloca[0, 0] [%7, 32] [1, 1] : memref<8x32xf32> to memref<?x32xf32, strided<[32, 1]>>
      %8 = vector.transfer_read %subview_6[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x32xf32, strided<[32, 1]>>, vector<8x32xf32>
      %9 = scf.for %arg2 = %c0 to %c384 step %c16 iter_args(%arg3 = %8) -> (vector<8x32xf32>) {
        %15 = vector.transfer_read %subview_5[%c0, %arg2], %cst : memref<?x384xf32, strided<[384, 1], offset: ?>>, vector<8x16xf32>
        %16 = vector.transfer_read %subview_4[%arg2, %arg1], %cst {in_bounds = [true, true]} : memref<384x128xf32, strided<[384, 1], offset: ?>>, vector<16x32xf32>
        %17 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %16, %arg3 : vector<8x16xf32>, vector<16x32xf32> into vector<8x32xf32>
        scf.yield %17 : vector<8x32xf32>
      }
      %subview_7 = memref.subview %subview_2[%arg0, %arg1] [%7, 32] [1, 1] : memref<?x128xf32, strided<[384, 1], offset: ?>> to memref<?x32xf32, strided<[384, 1], offset: ?>>
      vector.transfer_write %9, %subview_7[%c0, %c0] {in_bounds = [false, true]} : vector<8x32xf32>, memref<?x32xf32, strided<[384, 1], offset: ?>>
    }
  }
  return
}

// CHECK-LABEL:       func.func @matmul_scalar_load
// CHECK-COUNT-128:     memref.load

// -----

// Make sure we don't transpose a mask but create a transposed mask instead.

func.func @transpose_mask() {
  %a = arith.constant 4 : index
  %b = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<4x2xi1>
  %mask = vector.create_mask %a, %b : vector<2x4xi1>
  %transpose_mask = vector.transpose %mask, [1, 0] : vector<2x4xi1> to vector<4x2xi1>
  vector.transfer_write %transpose_mask, %3[%c0, %c0] {in_bounds = [true, true]} : vector<4x2xi1>, memref<4x2xi1>
  return
}

// CHECK-LABEL: func.func @transpose_mask
//   CHECK-NOT:   vector.create_mask
//   CHECK-NOT:   vector.constant_mask [2, 4]
//   CHECK-NOT:   vector.transpose
//   CHECK-NOT:   vector.shuffle
//       CHECK:   vector.constant_mask [4, 2] : vector<4x2xi1>

// -----

// Make sure that the gather patterns get rid of vector.gather over strided
// memref.

func.func @gather_strided_memref() {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %cst_0 = arith.constant dense<true> : vector<4xi1>
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2592000x3xf32, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %0, 64 : memref<2592000x3xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<518400xi32, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %1, 64 : memref<518400xi32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<518400xf32, #hal.descriptor_type<storage_buffer>>
  memref.assume_alignment %2, 64 : memref<518400xf32, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %0[0, 0] [2592000, 1] [1, 1] : memref<2592000x3xf32, #hal.descriptor_type<storage_buffer>> to memref<2592000xf32, strided<[3]>, #hal.descriptor_type<storage_buffer>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 4096)>()[%workgroup_id_x]
  %4 = affine.min affine_map<(d0) -> (-d0 + 518400, 4096)>(%3)
  %subview_1 = memref.subview %2[%3] [%4] [1] : memref<518400xf32, #hal.descriptor_type<storage_buffer>> to memref<?xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %subview_2 = memref.subview %1[%3] [%4] [1] : memref<518400xi32, #hal.descriptor_type<storage_buffer>> to memref<?xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %5 = affine.apply affine_map<()[s0] -> ((s0 floordiv 4) * 4)>()[%4]
  scf.for %arg0 = %c0 to %5 step %c4 {
    %6 = vector.transfer_read %subview_2[%arg0], %c0_i32 {in_bounds = [true]} : memref<?xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<4xi32>
    %7 = arith.index_cast %6 : vector<4xi32> to vector<4xindex>
    %8 = vector.gather %subview[%c0] [%7], %cst_0, %cst : memref<2592000xf32, strided<[3]>, #hal.descriptor_type<storage_buffer>>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
    vector.transfer_write %8, %subview_1[%arg0] {in_bounds = [true]} : vector<4xf32>, memref<?xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  }
  return
}

// CHECK-LABEL: func.func @gather_strided_memref
// CHECK-NOT: memref.subview {{.*}} : memref<2592000xf32, strided<[3]>
// CHECK-NOT: vector.gather %subview[%c0] [%7], %cst_0, %cst : memref<2592000xf32, strided<[3]>
