// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-vector-distribute{test-layout}, canonicalize, cse))' %s | FileCheck %s

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @mfma_matmul_96x64x16_mm(%lhs: vector<96x16xf16>, %rhs: vector<16x64xf16>, %init: vector<96x64xf32>) -> vector<96x64xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf32>
  return %0 : vector<96x64xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 32]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 2], outers_per_batch = [1, 1], threads_per_outer = [2, 32], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [32, 1]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [4, 1], threads_per_outer = [2, 32], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [32, 1]>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @mfma_matmul_96x64x16_mmt(%lhs: vector<96x16xf16>, %rhs: vector<64x16xf16>, %init: vector<96x64xf32>) -> vector<96x64xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, d2) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<64x16xf16> into vector<96x64xf32>
  return %0 : vector<96x64xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 32]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 32]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [4, 1], threads_per_outer = [2, 32], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [32, 1]>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @mfma_matmul_96x64x16_mmtt(%lhs: vector<96x16xf16>, %rhs: vector<64x16xf16>, %init: vector<64x96xf32>) -> vector<64x96xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (n, m)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<64x16xf16> into vector<64x96xf32>
  return %0 : vector<64x96xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME: subgroup_strides = [0, 0], thread_strides = [1, 32]
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME: subgroup_strides = [0, 0], thread_strides = [1, 32]
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 3], outers_per_batch = [1, 4], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME: subgroup_strides = [0, 0], thread_strides = [1, 32]

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 2, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 2, subgroup_n_count = 1>}>

func.func @matmul_192x64x16_mmt_multisubgroup(%lhs: vector<192x16xf16>, %rhs: vector<16x64xf16>, %init: vector<192x64xf32>) -> vector<192x64xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<192x16xf16>, vector<16x64xf16> into vector<192x64xf32>
  return %0 : vector<192x64xf32>
}

// CHECK: contract A vector layout: #iree_vector_ext.nested_layout<subgroups_per_workgroup = [2, 1]
// CHECK: contract B vector layout: #iree_vector_ext.nested_layout<subgroups_per_workgroup = [1, 1]
// CHECK: contract C vector layout: #iree_vector_ext.nested_layout<subgroups_per_workgroup = [2, 1]

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @matmul_16x16x256_read(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                 %rhs: memref<256x16xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                 %out: memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
  attributes { translation_info = #translation } {
  %alloc = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
  %alloc_0 = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
  %cst = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %5 = scf.for %arg0 = %c0 to %c256 step %c32 iter_args(%arg1 = %cst_1) -> (vector<16x16xf32>) {
    %6 = vector.transfer_read %lhs[%c0, %arg0], %cst {in_bounds = [true, true]} : memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<16x32xf16>
    %7 = vector.transfer_read %rhs[%arg0, %c0], %cst {in_bounds = [true, true]} : memref<256x16xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<32x16xf16>
    vector.transfer_write %6, %alloc_0[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    vector.transfer_write %7, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    %8 = vector.transfer_read %alloc_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<16x32xf16>
    %9 = vector.transfer_read %alloc[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<32x16xf16>
    %10 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %8, %9, %arg1 : vector<16x32xf16>, vector<32x16xf16> into vector<16x16xf32>
    scf.yield %10 : vector<16x16xf32>
  }
  vector.transfer_write %5, %out[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  memref.dealloc %alloc_0 : memref<16x32xf16, #gpu.address_space<workgroup>>
  memref.dealloc %alloc : memref<32x16xf16, #gpu.address_space<workgroup>>
  return
}

//      CHECK: transfer '{{.+}} memref<16x256xf16{{.+}}<storage_buffer>>, vector<16x32xf16>' vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 8],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [4, 1]>
//      CHECK: transfer '{{.+}} memref<256x16xf16{{.+}}<storage_buffer>>, vector<32x16xf16>' vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 8],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [2, 1]>

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 2], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 16]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [16, 1]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [16, 1]>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>


func.func @matmul_16x16x256_read_permute(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                         %rhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                         %out: memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
  attributes { translation_info = #translation } {
  %alloc = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
  %alloc_0 = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
  %cst = arith.constant 0.000000e+00 : f16
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %init_acc = vector.transfer_read %out[%c0, %c0], %cst_f32 {in_bounds = [true, true]}
      : memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<16x16xf32>
  %5 = scf.for %arg0 = %c0 to %c256 step %c32 iter_args(%arg1 = %init_acc) -> (vector<16x16xf32>) {
    %6 = vector.transfer_read %lhs[%c0, %arg0], %cst {in_bounds = [true, true]} : memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<16x32xf16>
    %7 = vector.transfer_read %rhs[%arg0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<32x16xf16>
    vector.transfer_write %6, %alloc_0[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    vector.transfer_write %7, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    %8 = vector.transfer_read %alloc_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<16x32xf16>
    %9 = vector.transfer_read %alloc[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<32x16xf16>
    %10 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %8, %9, %arg1 : vector<16x32xf16>, vector<32x16xf16> into vector<16x16xf32>
    scf.yield %10 : vector<16x16xf32>
  }
  vector.transfer_write %5, %out[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  memref.dealloc %alloc_0 : memref<16x32xf16, #gpu.address_space<workgroup>>
  memref.dealloc %alloc : memref<32x16xf16, #gpu.address_space<workgroup>>
  return
}

//  CHECK-NOT: transfer '{{.+}} memref<16x16xf16{{.+}}<storage_buffer>>, vector<16x16xf16>' vector layout
//      CHECK: transfer '{{.+}} memref<16x256xf16{{.+}}<storage_buffer>>, vector<16x32xf16>' vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 8],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [4, 1]>
//      CHECK: transfer '{{.+}} memref<16x256xf16{{.+}}storage_buffer>>, vector<32x16xf16>' vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [8, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 4]>

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 2], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 16]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [16, 1]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [16, 1]>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @matmul_16x16x256_fused(%lhs: memref<16x32xf16>,
                                  %rhs: memref<32x16xf16>,
                                  %bias: memref<16x16xf32>,
                                  %out: memref<16x16xf32>)
  attributes { translation_info = #translation } {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %acc = vector.transfer_read %out[%c0, %c0], %cst_f32 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %8 = vector.transfer_read %lhs[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x32xf16>, vector<16x32xf16>
  %9 = vector.transfer_read %rhs[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x16xf16>, vector<32x16xf16>
  %10 = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
    %8, %9, %acc : vector<16x32xf16>, vector<32x16xf16> into vector<16x16xf32>
  %11 = vector.transfer_read %bias[%c0, %c0], %cst_f32 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %12 = arith.addf %10, %11 : vector<16x16xf32>
  vector.transfer_write %12, %out[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  return
}

// We don't really care what layout we assign here, just that the only anchor
// we set is on the contraction.
//  CHECK-NOT: transfer {{.*}} vector layout
//      CHECK: contract A vector layout
//  CHECK-NOT: transfer {{.*}} vector layout
//      CHECK: contract B vector layout
//  CHECK-NOT: transfer {{.*}} vector layout
//      CHECK: contract C vector layout
//  CHECK-NOT: transfer {{.*}} vector layout

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [32, 1, 1]
                                              subgroup_size = 32,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @wmma_matmul_48x32x32_mm(%lhs: vector<48x32xf16>, %rhs: vector<32x32xf16>, %init: vector<48x32xf32>) -> vector<48x32xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<48x32xf16>, vector<32x32xf16> into vector<48x32xf32>
  return %0 : vector<48x32xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [1, 1], threads_per_outer = [16, 1], elements_per_thread = [1, 16],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 0]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 2], outers_per_batch = [1, 1], threads_per_outer = [1, 16], elements_per_thread = [16, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [0, 1]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [8, 1], threads_per_outer = [2, 16], elements_per_thread = [1, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [16, 1]>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [32, 1, 1]
                                              subgroup_size = 32,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @wmma_matmul_48x32x32_mmt(%lhs: vector<48x32xf16>, %rhs: vector<32x32xf16>, %init: vector<48x32xf32>) -> vector<48x32xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, d2) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<48x32xf16>, vector<32x32xf16> into vector<48x32xf32>
  return %0 : vector<48x32xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [1, 1], threads_per_outer = [16, 1], elements_per_thread = [1, 16],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 0]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 2], outers_per_batch = [1, 1], threads_per_outer = [16, 1], elements_per_thread = [1, 16],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [1, 0]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [8, 1], threads_per_outer = [2, 16], elements_per_thread = [1, 1],
// CHECK-SAME:   subgroup_strides = [0, 0], thread_strides = [16, 1]>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 2, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 1>}>


func.func @matmul_192x64x16_mmt_multi_m(%lhs: vector<2x64x16xf16>, %rhs: vector<16x64xf16>, %init: vector<2x64x64xf32>) -> vector<2x64x64xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<2x64x16xf16>, vector<16x64xf16> into vector<2x64x64xf32>
  return %0 : vector<2x64x64xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [2, 1, 1],
// CHECK-SAME:   batches_per_subgroup = [1, 4, 1],
// CHECK-SAME:   outers_per_batch = [1, 1, 1],
// CHECK-SAME:   threads_per_outer = [1, 16, 4],
// CHECK-SAME:   elements_per_thread = [1, 1, 4],
// CHECK-SAME:   subgroup_strides = [1, 0, 0],
// CHECK-SAME:   thread_strides = [0, 1, 16]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1],
// CHECK-SAME:   batches_per_subgroup = [1, 4],
// CHECK-SAME:   outers_per_batch = [1, 1],
// CHECK-SAME:   threads_per_outer = [4, 16],
// CHECK-SAME:   elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_strides = [0, 0],
// CHECK-SAME:   thread_strides = [16, 1]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [2, 1, 1],
// CHECK-SAME:   batches_per_subgroup = [1, 4, 4],
// CHECK-SAME:   outers_per_batch = [1, 1, 1],
// CHECK-SAME:   threads_per_outer = [1, 4, 16],
// CHECK-SAME:   elements_per_thread = [1, 4, 1],
// CHECK-SAME:   subgroup_strides = [1, 0, 0],
// CHECK-SAME:   thread_strides = [0, 16, 1]>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 2, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 4, subgroup_n_count = 1>}>

func.func @matmul_192x64x16_mmt_multi_split_m(%lhs: vector<2x64x16xf16>, %rhs: vector<16x64xf16>, %init: vector<2x64x64xf32>) -> vector<2x64x64xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<2x64x16xf16>, vector<16x64xf16> into vector<2x64x64xf32>
  return %0 : vector<2x64x64xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [2, 2, 1],
// CHECK-SAME:   batches_per_subgroup = [1, 2, 1],
// CHECK-SAME:   subgroup_strides = [2, 1, 0],
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [2, 2, 1],
// CHECK-SAME:   batches_per_subgroup = [1, 2, 4],
// CHECK-SAME:   subgroup_strides = [2, 1, 0],

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [128, 2, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule< intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>, workgroup_size = [128, 2, 1]}>

func.func @matmul_192x64x16_mmt_multi_m_and_n(%lhs: vector<4x64x16xf16>, %rhs: vector<2x16x64xf16>, %init: vector<4x2x64x64xf32>) -> vector<4x2x64x64xf32> attributes { translation_info = #translation } {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<4x64x16xf16>, vector<2x16x64xf16> into vector<4x2x64x64xf32>
  return %0 : vector<4x2x64x64xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [2, 1, 1],
// CHECK-SAME:   batches_per_subgroup = [2, 4, 1],
// CHECK-SAME:   subgroup_strides = [2, 0, 0],
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [2, 1, 1],
// CHECK-SAME:   batches_per_subgroup = [1, 1, 4],
// CHECK-SAME:   subgroup_strides = [1, 0, 0],
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [2, 2, 1, 1],
// CHECK-SAME:   batches_per_subgroup = [2, 1, 4, 4],
// CHECK-SAME:   subgroup_strides = [2, 1, 0, 0],

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [32, 4, 1]
                                              subgroup_size = 32,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4>}>

func.func @dequant_anchors_on_quant_only(%quant: memref<128x128xi4, strided<[4096, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                  %scale: memref<128xf16, strided<[32], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                  %zp: memref<128xf16, strided<[32], offset: ?>, #hal.descriptor_type<storage_buffer>>)
  attributes { translation_info = #translation } {
  %alloc = memref.alloc() : memref<128x128xf16, #gpu.address_space<workgroup>>
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %c0_i4 = arith.constant 0 : i4
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %quant[%c0, %c0], %c0_i4 {in_bounds = [true, true]} : memref<128x128xi4, strided<[4096, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<128x128xi4>
  %1 = vector.transfer_read %scale[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[32], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<128xf16>
  %2 = vector.broadcast %1 : vector<128xf16> to vector<128x128xf16>
  %3 = vector.transpose %2, [1, 0] : vector<128x128xf16> to vector<128x128xf16>
  %4 = vector.transfer_read %zp[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[32], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<128xf16>
  %5 = vector.broadcast %4 : vector<128xf16> to vector<128x128xf16>
  %6 = vector.transpose %5, [1, 0] : vector<128x128xf16> to vector<128x128xf16>
  %7 = arith.extui %0 : vector<128x128xi4> to vector<128x128xi32>
  %8 = arith.uitofp %7 : vector<128x128xi32> to vector<128x128xf16>
  %9 = arith.subf %8, %6 : vector<128x128xf16>
  %10 = arith.mulf %9, %3 : vector<128x128xf16>
  vector.transfer_write %10, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
  return
}
//      CHECK: transfer '{{.+}} memref<128x128xi4{{.+}}<storage_buffer>>, vector<128x128xi4>' vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [4, 1], outers_per_batch = [1, 1], threads_per_outer = [32, 4], elements_per_thread = [1, 32], subgroup_strides = [0, 0], thread_strides = [4, 1]>
//  CHECK-NOT: transfer '{{.+}} memref<128xf16{{.+}}<storage_buffer>>, vector<128xf16>' vector layout

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [128, 2, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>
func.func @batch_matmul_unit_batch(%arg0: vector<1x64x64xf16>, %arg1: vector<1x64x128xf16>, %arg2: vector<1x64x128xf32>) -> vector<1x64x128xf32> attributes {translation_info = #translation} {
  %0 = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %arg0, %arg1, %arg2 : vector<1x64x64xf16>, vector<1x64x128xf16> into vector<1x64x128xf32>
  return %0 : vector<1x64x128xf32>
}
//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 2, 1],
// CHECK-SAME:   batches_per_subgroup = [1, 2, 4],
// CHECK-SAME:   outers_per_batch = [1, 1, 1]
// CHECK-SAME:   threads_per_outer = [1, 16, 4]
// CHECK-SAME:   elements_per_thread = [1, 1, 4]
// CHECK-SAME:   subgroup_strides = [0, 2, 0],
// CHECK-SAME:   thread_strides = [0, 1, 16]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1, 2]
// CHECK-SAME:   batches_per_subgroup = [1, 4, 4]
// CHECK-SAME:   outers_per_batch = [1, 1, 1]
// CHECK-SAME:   threads_per_outer = [1, 4, 16]
// CHECK-SAME:   elements_per_thread = [1, 4, 1]
// CHECK-SAME:   subgroup_strides = [0, 0, 1],
// CHECK-SAME:   thread_strides = [0, 16, 1]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 2, 2]
// CHECK-SAME:   batches_per_subgroup = [1, 2, 4]
// CHECK-SAME:   outers_per_batch = [1, 1, 1]
// CHECK-SAME:   threads_per_outer = [1, 4, 16]
// CHECK-SAME:   elements_per_thread = [1, 4, 1]
// CHECK-SAME:   subgroup_strides = [0, 2, 1],
// CHECK-SAME:   thread_strides = [0, 16, 1]>
