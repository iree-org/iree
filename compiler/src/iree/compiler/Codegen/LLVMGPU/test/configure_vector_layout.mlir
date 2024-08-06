// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-configure-vector-layouts, canonicalize, cse))' %s | FileCheck %s

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1>}>

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 8], subgroup_strides = [0, 0], thread_strides = [4, 1]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [32, 2], elements_per_thread = [1, 8], subgroup_strides = [0, 0], thread_strides = [2, 1]>

// CHECK-LABEL: func.func @matmul_16x16x256_read
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
    // CHECK: %[[READ0:.+]] = vector.transfer_read
    // CHECK: to_layout %[[READ0]] to #[[$NESTED]]
    // CHECK: %[[READ1:.+]] = vector.transfer_read
    // CHECK: to_layout %[[READ1]] to #[[$NESTED1]]
    vector.transfer_write %6, %alloc_0[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    vector.transfer_write %7, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    %8 = vector.transfer_read %alloc_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<16x32xf16>
    %9 = vector.transfer_read %alloc[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<32x16xf16>
    // CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout
    // CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout
    // CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout
    // CHECK: vector.contract
    // CHECK-SAME: %[[LHS]], %[[RHS]], %[[ACC]]
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

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1>}>

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [16, 4], elements_per_thread = [1, 8], subgroup_strides = [0, 0], thread_strides = [4, 1]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [1, 1], outers_per_batch = [1, 1], threads_per_outer = [4, 16], elements_per_thread = [8, 1], subgroup_strides = [0, 0], thread_strides = [1, 4]>

// CHECK-LABEL: func.func @matmul_16x16x256_read_permute
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
  // CHECK: scf.for
  %5 = scf.for %arg0 = %c0 to %c256 step %c32 iter_args(%arg1 = %init_acc) -> (vector<16x16xf32>) {
    %6 = vector.transfer_read %lhs[%c0, %arg0], %cst {in_bounds = [true, true]} : memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<16x32xf16>
    %7 = vector.transfer_read %rhs[%arg0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<32x16xf16>
    // CHECK: %[[READ0:.+]] = vector.transfer_read
    // CHECK: to_layout %[[READ0]] to #[[$NESTED]]
    // CHECK: %[[READ1:.+]] = vector.transfer_read
    // CHECK: to_layout %[[READ1]] to #[[$NESTED1]]
    vector.transfer_write %6, %alloc_0[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    vector.transfer_write %7, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    %8 = vector.transfer_read %alloc_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<16x32xf16>
    %9 = vector.transfer_read %alloc[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<32x16xf16>
    // CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout
    // CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout
    // CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout
    // CHECK: vector.contract
    // CHECK-SAME: %[[LHS]], %[[RHS]], %[[ACC]]
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

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1>}>

// We don't really care what layout we assign here, just that the only anchor
// we set is on the contraction.

// CHECK-LABEL: func.func @matmul_16x16x256_fused
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
  // CHECK-DAG: %[[READA:.+]] = vector.transfer_read
  // CHECK-DAG: %[[READB:.+]] = vector.transfer_read
  // CHECK-DAG: %[[READC:.+]] = vector.transfer_read
  // CHECK-NOT: to_layout %[[READA]]
  // CHECK-NOT: to_layout %[[READB]]
  // CHECK-NOT: to_layout %[[READC]]

  // CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout
  // CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout
  // CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout
  // CHECK: vector.contract
  // CHECK-SAME: %[[LHS]], %[[RHS]], %[[ACC]]
  %10 = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
    %8, %9, %acc : vector<16x32xf16>, vector<32x16xf16> into vector<16x16xf32>
  %11 = vector.transfer_read %bias[%c0, %c0], %cst_f32 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
  %12 = arith.addf %10, %11 : vector<16x16xf32>
  vector.transfer_write %12, %out[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
  return
}

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [32, 4, 1]
                                              subgroup_size = 32,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4>}>

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [4, 1], outers_per_batch = [1, 1], threads_per_outer = [32, 4], elements_per_thread = [1, 32], subgroup_strides = [0, 0], thread_strides = [4, 1]>

// CHECK-LABEL: func.func @dequant_anchors_on_quant_only
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
  // CHECK: %[[READ:.+]] = vector.transfer_read
  // CHECK: to_layout %[[READ]] to #[[$NESTED]]
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
