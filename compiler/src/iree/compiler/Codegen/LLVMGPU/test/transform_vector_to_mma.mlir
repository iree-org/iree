// RUN: iree-opt %s --split-input-file -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<32x32xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<32x32xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<32x32xf32>
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
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
    } : !transform.any_op
    transform.iree.vector.vector_to_mma_conversion %func { use_wmma } : (!transform.any_op) -> ()

    // Apply canonicalization post-hoc to trigger DCE and pass the test
    // (i.e. all vector.contract are dead).
    // TODO: consider having the vector_to_mma_conversion do the DCE automatically.
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  } // @__transform_main
} // module

// -----

// Verify that unrolling does not apply to rank 1 elementwise vector ops.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @gathered_matmul  {
builtin.module {
// CHECK-LABEL: func.func @gathered_matmul
func.func @gathered_matmul() {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %cst_mask = arith.constant dense<true> : vector<4x4xi1>
  %cst_pt = arith.constant dense<0.000000e+00> : vector<4x4xf32>
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
  %cst_2 = arith.constant dense<1> : vector<4x4xindex>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<32x32xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<32x32xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<32x32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
  %3 = gpu.thread_id  x
  %4 = gpu.thread_id  y
  %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%4]
  %6 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%3]
// CHECK: gpu.subgroup_mma_constant_matrix %{{.*}} : !gpu.mma_matrix<16x16xf32, "COp">
// CHECK: scf.for {{.*}} -> (!gpu.mma_matrix<16x16xf32, "COp">) {
// CHECK:   arith.addi {{.*}} : vector<4xindex>
// CHECK:   vector.gather {{.*}} : memref<32x32xf32>, vector<4x4xindex>, vector<4x4xi1>, vector<4x4xf32> into vector<4x4xf32>
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
    %10 = vector.broadcast %arg0 : index to vector<4xindex>
    %11 = arith.addi %10, %cst_1 : vector<4xindex>
    %12 = vector.broadcast %11 : vector<4xindex> to vector<4x4xindex>
    %13 = arith.addi %12, %cst_2 : vector<4x4xindex>
    %14 = vector.gather %0[%c0, %c0] [%13], %cst_mask, %cst_pt : memref<32x32xf32>, vector<4x4xindex>, vector<4x4xi1>, vector<4x4xf32> into vector<4x4xf32>
    vector.transfer_write %14, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32>
    gpu.barrier
    %15 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
    %16 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %17 = vector.transfer_read %alloc[%15, %16], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %18 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
    %19 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%arg0]
    %20 = vector.transfer_read %1[%19, %18], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x16xf32>
    %21 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %20, %arg1 : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
    scf.yield %21 : vector<16x16xf32>
  }
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%5]
  %9 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%c0)[%6]
  vector.transfer_write %7, %2[%8, %9] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32>
  return
}
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
  } : !transform.any_op
  transform.iree.vector.vector_to_mma_conversion %func { use_wmma } : (!transform.any_op) -> ()
  transform.apply_patterns to %func {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
    transform.yield
  } // @__transform_main
} // module
}
