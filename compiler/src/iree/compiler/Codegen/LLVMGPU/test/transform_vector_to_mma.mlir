// RUN: iree-opt %s --split-input-file -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

hal.executable private @matmul  {
builtin.module {
// CHECK-LABEL: func.func @matmul
func.func @matmul() {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf32>
  memref.assume_alignment %0, 64 : memref<32x32xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf32>
  memref.assume_alignment %1, 64 : memref<32x32xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf32>
  memref.assume_alignment %2, 64 : memref<32x32xf32>
  %3 = gpu.thread_id  x
  %4 = gpu.thread_id  y
  %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%4]
  %6 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%3]
// CHECK: gpu.subgroup_mma_constant_matrix %{{.*}} : !gpu.mma_matrix<16x16xf32, "COp">
// CHECK: scf.for {{.*}} -> (!gpu.mma_matrix<16x16xf32, "COp">) {
// CHECK:   gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 32 : index} : memref<32x32xf32> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:   gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 32 : index} : memref<32x32xf32> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:   gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 32 : index} : memref<32x32xf32> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:   gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 32 : index} : memref<32x32xf32> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:   gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:   gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:   scf.yield {{.*}} : !gpu.mma_matrix<16x16xf32, "COp">
// CHECK: }
// CHECK: gpu.subgroup_mma_store_matrix {{.*}} {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<32x32xf32>
  %7 = scf.for %arg0 = %c0 to %c32 step %c16 iter_args(%arg1 = %cst) -> (vector<16x16xf32>) {
    %10 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
    %11 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %12 = vector.transfer_read %0[%10, %11], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %16 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %18 = vector.transfer_read %1[%17, %16], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %18, %arg1 : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
    scf.yield %22 : vector<16x16xf32>
  }
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
  %9 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
  vector.transfer_write %7, %2[%8, %9] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32>
  return
}
}
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func { unroll_vectors_gpu_wmma } : (!pdl.operation) -> ()
  transform.iree.vector.vector_to_mma_conversion %func { use_wmma } : (!pdl.operation) -> ()

  // Apply canonicalization post-hoc to trigger DCE and pass the test 
  // (i.e. all vector.contract are dead).
  // TODO: consider having the vector_to_mma_conversion do the DCE automatically.
  transform.iree.apply_patterns %func { canonicalization } : (!pdl.operation) -> ()
}
}

// -----

#map = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (-s0 + 2)>
#map2 = affine_map<()[s0] -> (s0 * -32 + 1024)>
#map3 = affine_map<()[s0] -> (s0 * -32 + 64)>
#map4 = affine_map<()[s0] -> (s0 mod 8)>
#map5 = affine_map<()[s0, s1] -> (((s0 + s1 * 8) mod 32) floordiv 8)>
#map6 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
#map7 = affine_map<()[s0, s1] -> ((((s0 + s1 * 8) mod 32) floordiv 8) * 4)>
#map8 = affine_map<()[s0] -> (s0 * -4 + (s0 floordiv 8) * 32 + 32)>
#map9 = affine_map<()[s0, s1] -> ((((s0 + s1 * 8) mod 32) floordiv 8) * -4 + 16)>
#map10 = affine_map<()[s0] -> (-s0 + 64)>
#map11 = affine_map<()[s0] -> (-s0 + 576)>
#map12 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map13 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d1)>
hal.executable private @convolution  {
builtin.module {
// CHECK-LABEL: func.func @convolution
func.func @convolution() {
  %cst = arith.constant dense<0.000000e+00> : vector<1x32x32xf16>
  %cst_0 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
  %cst_1 = arith.constant dense<-1> : vector<4x4xindex>
  %cst_2 = arith.constant dense<0> : vector<4x4xindex>
  %cst_3 = arith.constant dense<32> : vector<4x4xindex>
  %c4 = arith.constant 4 : index
  %cst_4 = arith.constant dense<192> : vector<4xindex>
  %cst_5 = arith.constant dense<0> : vector<4xindex>
  %cst_6 = arith.constant dense<64> : vector<4xindex>
  %cst_7 = arith.constant dense<-1> : vector<4xindex>
  %cst_8 = arith.constant dense<true> : vector<4x4xi1>
  %cst_9 = arith.constant dense<0.000000e+00> : vector<4x4xf16>
  %cst_10 = arith.constant dense<34> : vector<4x4xindex>
  %cst_11 = arith.constant dense<64> : vector<4x4xindex>
  %c0 = arith.constant 0 : index
  %cst_12 = arith.constant 0.000000e+00 : f16
  %c16 = arith.constant 16 : index
  %c576 = arith.constant 576 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x32xf16>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2x34x34x64xf16>
  memref.assume_alignment %0, 64 : memref<2x34x34x64xf16>
  %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<2x1024x64xf16>
  memref.assume_alignment %1, 64 : memref<2x1024x64xf16>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<576x64xf16>
  memref.assume_alignment %2, 64 : memref<576x64xf16>
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %3 = affine.apply #map()[%workgroup_id_x]
  %4 = affine.apply #map()[%workgroup_id_y]
  %5 = affine.apply #map1()[%workgroup_id_z]
  %6 = affine.apply #map2()[%workgroup_id_y]
  %7 = affine.apply #map3()[%workgroup_id_x]
  %subview = memref.subview %1[%workgroup_id_z, %4, %3] [%5, %6, %7] [1, 1, 1] : memref<2x1024x64xf16> to memref<?x?x?xf16, strided<[65536, 64, 1], offset: ?>>
  vector.transfer_write %cst, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x32x32xf16>, memref<?x?x?xf16, strided<[65536, 64, 1], offset: ?>>
  gpu.barrier
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x32x16xf16>
  %8 = gpu.thread_id  x
  %9 = gpu.thread_id  y
  %10 = affine.apply #map4()[%8]
  %11 = affine.apply #map5()[%8, %9]
  %12 = affine.apply #map6()[%8]
  %13 = affine.apply #map7()[%8, %9]
  %14 = vector.broadcast %cst_0 : vector<4xindex> to vector<4x4xindex>
  %15 = vector.transpose %14, [1, 0] : vector<4x4xindex> to vector<4x4xindex>
  %16 = arith.muli %11, %c4 : index
  %17 = vector.broadcast %16 : index to vector<4xindex>
  %18 = arith.muli %10, %c4 : index
  %19 = vector.broadcast %18 : index to vector<4x4xindex>
  %20 = arith.addi %15, %19 : vector<4x4xindex>
  %21 = arith.cmpi slt, %20, %cst_2 : vector<4x4xindex>
  %22 = arith.subi %cst_1, %20 : vector<4x4xindex>
  %23 = arith.select %21, %22, %20 : vector<4x4xi1>, vector<4x4xindex>
  %24 = arith.divsi %23, %cst_3 : vector<4x4xindex>
  %25 = arith.subi %cst_1, %24 : vector<4x4xindex>
  %26 = arith.select %21, %25, %24 : vector<4x4xi1>, vector<4x4xindex>
  %27 = vector.broadcast %workgroup_id_y : index to vector<4x4xindex>
  %28 = arith.addi %27, %26 : vector<4x4xindex>
  %29 = arith.remsi %20, %cst_3 : vector<4x4xindex>
  %30 = arith.cmpi slt, %29, %cst_2 : vector<4x4xindex>
  %31 = arith.addi %29, %cst_3 : vector<4x4xindex>
  %32 = arith.select %30, %31, %29 : vector<4x4xi1>, vector<4x4xindex>
  %33 = vector.broadcast %workgroup_id_z : index to vector<4x4xindex>
  %34 = arith.muli %33, %cst_10 : vector<4x4xindex>
  %35 = arith.addi %12, %3 : index
  %36 = affine.apply #map8()[%8]
  %37 = affine.apply #map9()[%8, %9]
  %subview_14 = memref.subview %alloc_13[0, %12, %13] [1, %36, %37] [1, 1, 1] : memref<1x32x16xf16> to memref<1x?x?xf16, strided<[512, 16, 1], offset: ?>>
  %38 = affine.apply #map10()[%35]
  %subview_15 = memref.subview %alloc[%13, %12] [%37, %36] [1, 1] : memref<16x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
// CHECK:         scf.for {{.*}} {
// CHECK:           vector.gather {{.*}} : memref<2x34x34x64xf16>, vector<4x4xindex>, vector<4x4xi1>, vector<4x4xf16> into vector<4x4xf16>
// CHECK-COUNT-2:   gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 16 : index} : memref<1x32x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-COUNT-2:   gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 32 : index} : memref<16x32xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-COUNT-4:   gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 64 : index} : memref<?x?x?xf16, strided<[65536, 64, 1], offset: ?>> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-COUNT-4:   gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-COUNT-4:   gpu.subgroup_mma_store_matrix {{.*}} {leadDimension = 64 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<?x?x?xf16, strided<[65536, 64, 1], offset: ?>>
// CHECK:         }
  scf.for %arg0 = %c0 to %c576 step %c16 {
    %39 = vector.broadcast %arg0 : index to vector<4xindex>
    %40 = arith.addi %39, %cst_0 : vector<4xindex>
    %41 = arith.addi %40, %17 : vector<4xindex>
    %42 = arith.remsi %41, %cst_6 : vector<4xindex>
    %43 = arith.cmpi slt, %42, %cst_5 : vector<4xindex>
    %44 = arith.addi %42, %cst_6 : vector<4xindex>
    %45 = arith.select %43, %44, %42 : vector<4xi1>, vector<4xindex>
    %46 = arith.cmpi slt, %41, %cst_5 : vector<4xindex>
    %47 = arith.subi %cst_7, %41 : vector<4xindex>
    %48 = arith.select %46, %47, %41 : vector<4xi1>, vector<4xindex>
    %49 = arith.divsi %48, %cst_4 : vector<4xindex>
    %50 = arith.subi %cst_7, %49 : vector<4xindex>
    %51 = arith.select %46, %50, %49 : vector<4xi1>, vector<4xindex>
    %52 = vector.broadcast %51 : vector<4xindex> to vector<4x4xindex>
    %53 = arith.addi %28, %52 : vector<4x4xindex>
    %54 = arith.remsi %41, %cst_4 : vector<4xindex>
    %55 = arith.cmpi slt, %54, %cst_5 : vector<4xindex>
    %56 = arith.addi %54, %cst_4 : vector<4xindex>
    %57 = arith.select %55, %56, %54 : vector<4xi1>, vector<4xindex>
    %58 = arith.cmpi slt, %57, %cst_5 : vector<4xindex>
    %59 = arith.subi %cst_7, %57 : vector<4xindex>
    %60 = arith.select %58, %59, %57 : vector<4xi1>, vector<4xindex>
    %61 = arith.divsi %60, %cst_6 : vector<4xindex>
    %62 = arith.subi %cst_7, %61 : vector<4xindex>
    %63 = arith.select %58, %62, %61 : vector<4xi1>, vector<4xindex>
    %64 = vector.broadcast %63 : vector<4xindex> to vector<4x4xindex>
    %65 = arith.addi %32, %64 : vector<4x4xindex>
    %66 = arith.addi %53, %34 : vector<4x4xindex>
    %67 = arith.muli %66, %cst_10 : vector<4x4xindex>
    %68 = arith.addi %65, %67 : vector<4x4xindex>
    %69 = arith.muli %68, %cst_11 : vector<4x4xindex>
    %70 = vector.broadcast %45 : vector<4xindex> to vector<4x4xindex>
    %71 = arith.addi %70, %69 : vector<4x4xindex>
    %72 = vector.gather %0[%c0, %c0, %c0, %c0] [%71], %cst_8, %cst_9 : memref<2x34x34x64xf16>, vector<4x4xindex>, vector<4x4xi1>, vector<4x4xf16> into vector<4x4xf16>
    vector.transfer_write %72, %subview_14[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<4x4xf16>, memref<1x?x?xf16, strided<[512, 16, 1], offset: ?>>
    gpu.barrier
    %73 = arith.addi %13, %arg0 : index
    %74 = affine.apply #map11()[%73]
    %subview_16 = memref.subview %2[%73, %35] [%74, %38] [1, 1] : memref<576x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    %75 = vector.transfer_read %subview_16[%c0, %c0], %cst_12 {in_bounds = [true, true]} : memref<?x?xf16, strided<[64, 1], offset: ?>>, vector<4x4xf16>
    vector.transfer_write %75, %subview_15[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf16>, memref<?x?xf16, strided<[32, 1], offset: ?>>
    gpu.barrier
    %76 = vector.transfer_read %alloc_13[%c0, %c0, %c0], %cst_12 {in_bounds = [true, true]} : memref<1x32x16xf16>, vector<32x16xf16>
    %77 = vector.transfer_read %alloc[%c0, %c0], %cst_12 {in_bounds = [true, true]} : memref<16x32xf16>, vector<16x32xf16>
    %78 = vector.transfer_read %subview[%c0, %c0, %c0], %cst_12 {in_bounds = [true, true]} : memref<?x?x?xf16, strided<[65536, 64, 1], offset: ?>>, vector<32x32xf16>
    %79 = vector.contract {indexing_maps = [#map12, #map13, #map14], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %76, %77, %78 : vector<32x16xf16>, vector<16x32xf16> into vector<32x32xf16>
    vector.transfer_write %79, %subview[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<32x32xf16>, memref<?x?x?xf16, strided<[65536, 64, 1], offset: ?>>
    gpu.barrier
  }
  memref.dealloc %alloc_13 : memref<1x32x16xf16>
  memref.dealloc %alloc : memref<16x32xf16>
  return
}
}
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func { unroll_vectors_gpu_wmma } : (!pdl.operation) -> ()
  transform.iree.vector.vector_to_mma_conversion %func { use_wmma } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { canonicalization } : (!pdl.operation) -> ()
}
}
