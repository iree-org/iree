// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-vector-distribute{test-layout}, canonicalize, cse))' %s | FileCheck %s

func.func @matmul_96x64x16_mm(%lhs: vector<96x16xf16>, %rhs: vector<16x64xf16>, %init: vector<96x64xf32>) -> vector<96x64xf32> attributes {
    mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
      subgroup_m_count = 1, subgroup_n_count = 1, subgroup_m_tile_count = 3, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>,
    workgroup_size = [64, 1, 1]} {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf32>
  return %0 : vector<96x64xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [1, 0], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [2, 32]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 2], outers_per_batch = [1, 1], threads_per_outer = [2, 32], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [2, 32]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [4, 1], threads_per_outer = [2, 32], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [2, 32]>

// -----

func.func @matmul_96x64x16_mmt(%lhs: vector<96x16xf16>, %rhs: vector<64x16xf16>, %init: vector<96x64xf32>) -> vector<96x64xf32> attributes {
    mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
      subgroup_m_count = 1, subgroup_n_count = 1, subgroup_m_tile_count = 3, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>,
    workgroup_size = [64, 1, 1]} {
    %0 = vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, d2) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<64x16xf16> into vector<96x64xf32>
  return %0 : vector<96x64xf32>
}

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [1, 0], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [2, 32]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 2], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_order = [1, 0], batch_order = [1, 0], outer_order = [1, 0], thread_order = [1, 0], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [2, 32]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [3, 2], outers_per_batch = [4, 1], threads_per_outer = [2, 32], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [2, 32]>

// -----

func.func @matmul_192x64x16_mmt_multisubgroup(%lhs: vector<192x16xf16>, %rhs: vector<16x64xf16>, %init: vector<192x64xf32>) -> vector<192x64xf32> attributes {
    mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
      subgroup_m_count = 2, subgroup_n_count = 1, subgroup_m_tile_count = 3, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>,
    workgroup_size = [64, 2, 1]} {
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

func.func @matmul_16x16x256_read(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                 %rhs: memref<256x16xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                 %out: memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
  attributes {
    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
                     subgroup_m_count = 1, subgroup_n_count = 1, subgroup_m_tile_count = 1, subgroup_n_tile_count = 1, subgroup_k_tile_count = 2>,
    workgroup_size = [64, 1, 1]} {
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
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [16, 4]>
//      CHECK: transfer '{{.+}} memref<256x16xf16{{.+}}<storage_buffer>>, vector<32x16xf16>' vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 8],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [32, 2]>

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 2], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [1, 0], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [4, 16]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [4, 16]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [4, 16]>

// -----

func.func @matmul_16x16x256_read_permute(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                         %rhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                         %out: memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
  attributes {
    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
                     subgroup_m_count = 1, subgroup_n_count = 1, subgroup_m_tile_count = 1, subgroup_n_tile_count = 1, subgroup_k_tile_count = 2>,
    workgroup_size = [64, 1, 1]} {
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
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [16, 4]>
//      CHECK: transfer '{{.+}} memref<16x256xf16{{.+}}storage_buffer>>, vector<32x16xf16>' vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [8, 1],
// CHECK-SAME:   subgroup_order = [1, 0], batch_order = [1, 0], outer_order = [1, 0], thread_order = [1, 0], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [4, 16]>

//      CHECK: contract A vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 2], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 4],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [1, 0], element_order = [0, 1],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [4, 16]>
//      CHECK: contract B vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [4, 16]>
//      CHECK: contract C vector layout: #iree_vector_ext.nested_layout<
// CHECK-SAME:   subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [4, 1],
// CHECK-SAME:   subgroup_order = [0, 1], batch_order = [0, 1], outer_order = [0, 1], thread_order = [0, 1], element_order = [1, 0],
// CHECK-SAME:   subgroup_basis = [1, 1], thread_basis = [4, 16]>

