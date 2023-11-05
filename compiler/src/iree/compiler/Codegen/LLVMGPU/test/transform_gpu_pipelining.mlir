// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

hal.executable private @matmul_pipelining  {
builtin.module {
func.func @matmul_pipelining() {
  %c2048 = arith.constant 2048 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf16, "COp">
  %1 = gpu.thread_id  x
  %2 = gpu.thread_id  y
  %3 = gpu.thread_id  z
  %4 = memref.alloc() : memref<4x32x40xf16, 3>
  %5 = memref.alloc() : memref<4x32x40xf16, 3>
  %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<3456x2048xf16>
  memref.assume_alignment %6, 64 : memref<3456x2048xf16>
  %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2048x1024xf16>
  memref.assume_alignment %7, 64 : memref<2048x1024xf16>
  %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<3456x1024xf16>
  memref.assume_alignment %8, 64 : memref<3456x1024xf16>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %9 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%1, %2, %3]
  %10 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%1]
  %11 = scf.for %arg0 = %c0 to %c2048 step %c32 iter_args(%arg1 = %0) -> (!gpu.mma_matrix<16x16xf16, "COp">) {
    gpu.barrier
    %14 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s1 floordiv 4) * 32)>()[%arg0, %1]
    %15 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 * 32 + s3 * 32 + s0 floordiv 4)>()[%1, %2, %3, %workgroup_id_y]
    %16 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 4)>(%arg0)
    %17 = nvgpu.device_async_copy %6[%15, %14], %4[%16, %9, %10], 8 : memref<3456x2048xf16> to memref<4x32x40xf16, 3>
    %18 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 16 + s3 * 32 + s1 floordiv 4)>()[%arg0, %1, %2, %3]
    %19 = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32 - (s0 floordiv 4) * 32)>()[%1, %workgroup_id_x]
    %20 = nvgpu.device_async_copy %7[%18, %19], %5[%16, %9, %10], 8 : memref<2048x1024xf16> to memref<4x32x40xf16, 3>
    %21 = nvgpu.device_async_create_group %17, %20
    nvgpu.device_async_wait %21
    gpu.barrier
    %22 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%2]
    %23 = gpu.subgroup_mma_load_matrix %4[%16, %22, %c0] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %24 = gpu.subgroup_mma_load_matrix %4[%16, %22, %c16] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %25 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%1]
    %26 = gpu.subgroup_mma_load_matrix %5[%16, %c0, %25] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %27 = gpu.subgroup_mma_load_matrix %5[%16, %c16, %25] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %28 = gpu.subgroup_mma_compute %23, %26, %arg1 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
    %29 = gpu.subgroup_mma_compute %24, %27, %28 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
    scf.yield %29 : !gpu.mma_matrix<16x16xf16, "COp">
  }
  %12 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 32)>()[%2, %workgroup_id_y]
  %13 = affine.apply affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 32) * 16)>()[%1, %workgroup_id_x]
  gpu.subgroup_mma_store_matrix %11, %8[%12, %13] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<3456x1024xf16>
  return
}
}
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %for = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %for : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.iree.pipeline_shared_memory_copies %1 { depth = 4 } : (!transform.op<"scf.for">) -> !transform.op<"scf.for">
    transform.yield
  } // @__transform_main
} // module

// CHECK-LABEL: func.func @matmul_pipelining
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_create_group
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_create_group
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_create_group
// CHECK: scf.for
// CHECK:   nvgpu.device_async_wait %{{.*}} {numGroups = 3 : i32}
// CHECK:   nvgpu.device_async_copy
// CHECK:   nvgpu.device_async_copy
// CHECK:   nvgpu.device_async_create_group

