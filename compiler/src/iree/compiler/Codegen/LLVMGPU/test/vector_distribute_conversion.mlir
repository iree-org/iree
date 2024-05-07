// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmgpu-vector-distribute, canonicalize, cse))' -split-input-file %s | FileCheck %s

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute 
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64, 
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @mfma_matmul_256x256x256(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
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

// CHECK-LABEL: func.func @mfma_matmul_256x256x256
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x1x1x4x1xf32>
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}} = %[[INIT]])
//       CHECK:     %[[LLOAD:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[RLOAD:.+]] = vector.transfer_read {{.*}} : memref<256x16xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     vector.transfer_write %[[LLOAD]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD]], %[[RHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     gpu.barrier
// CHECK-COUNT-2:   vector.transfer_read %[[LHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<1x4xf16>
// CHECK-COUNT-2:   vector.transfer_read %[[RHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
// CHECK-COUNT-2:   amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:     %[[BCAST:.+]] = vector.broadcast {{.*}} : vector<1x1x4x1xf32> to vector<1x1x1x1x4x1xf32>
//       CHECK:     scf.yield %[[BCAST]]
//       CHECK:  vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<16x16xf32{{.*}}>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute 
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64, 
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @mfma_matmul_256x256x256(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
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

// CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 64)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @mfma_matmul_256x256x256
//       CHECK:   %[[TIDX:.+]] = gpu.thread_id  x
//       CHECK:   %[[TIDY:.+]] = gpu.thread_id  y
//       CHECK:   %[[TIDZ:.+]] = gpu.thread_id  z
//       CHECK:   %[[LIN_ID:.+]] = affine.apply #[[$MAP]]()[%[[TIDX]], %[[TIDY]], %[[TIDZ]]]
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:   iree_vector_ext.thread_ids %[[LIN_ID]]
//       CHECK:   %[[INIT_READ:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<4x1xf32>
//       CHECK:   %[[INIT:.+]] = vector.insert_strided_slice %[[INIT_READ]]
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}} = %[[INIT]]) -> (vector<1x1x1x1x4x1xf32>)
//       CHECK:     %[[LLOAD:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[RLOAD:.+]] = vector.transfer_read {{.*}} permutation_map = #[[$MAP1]]} : memref<16x256xf16, {{.*}}>, vector<8x1xf16>
//       CHECK:     vector.transfer_write %[[LLOAD]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD]], %[[RHS_ALLOC]]{{.*}} : vector<8x1xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     gpu.barrier
// CHECK-COUNT-2:   vector.transfer_read %[[LHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<1x4xf16>
// CHECK-COUNT-2:   vector.transfer_read %[[RHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
// CHECK-COUNT-2:   amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:     %[[BCAST:.+]] = vector.broadcast {{.*}} : vector<1x1x4x1xf32> to vector<1x1x1x1x4x1xf32>
//       CHECK:     scf.yield %[[BCAST]]
//       CHECK:  vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<16x16xf32{{.*}}>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute 
                                              workgroup_size = [32, 1, 1]
                                              subgroup_size = 32, 
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @wmma_matmul_256x256x256(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
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

// CHECK-LABEL: func.func @wmma_matmul_256x256x256
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x8x1x1x1xf32>
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}} = %[[INIT]]) -> (vector<1x1x8x1x1x1xf32>)
//       CHECK:     %[[LLOAD0:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[LLOAD1:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[RLOAD0:.+]] = vector.transfer_read {{.*}} : memref<256x16xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[RLOAD1:.+]] = vector.transfer_read {{.*}} : memref<256x16xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     vector.transfer_write %[[LLOAD0]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[LLOAD1]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD0]], %[[RHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD1]], %[[RHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     gpu.barrier
// CHECK-COUNT-2:   vector.transfer_read %[[LHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<1x16xf16>
// CHECK-COUNT-2:   vector.transfer_read %[[RHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<16x1xf16>
// CHECK-COUNT-2:   amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<8xf32>
//       CHECK:     %[[BCAST:.+]] = vector.broadcast {{.*}} : vector<8x1x1x1xf32> to vector<1x1x8x1x1x1xf32>
//       CHECK:     scf.yield %[[BCAST]] : vector<1x1x8x1x1x1xf32>
// CHECK-COUNT-8:  vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<1x1xf32>, memref<16x16xf32{{.*}}>

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute 
                                              workgroup_size = [32, 1, 1]
                                              subgroup_size = 32, 
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 1>}>

func.func @wmma_matmul_256x256x256(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
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
// TODO: Currently have many inits because of the strided/interleaved layout of the C/D matrix.
//       Need to add some canonicalization pattern to improve this.

// CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 32)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @wmma_matmul_256x256x256
//       CHECK:   %[[TIDX:.+]] = gpu.thread_id  x
//       CHECK:   %[[TIDY:.+]] = gpu.thread_id  y
//       CHECK:   %[[TIDZ:.+]] = gpu.thread_id  z
//       CHECK:   %[[LIN_ID:.+]] = affine.apply #[[$MAP]]()[%[[TIDX]], %[[TIDY]], %[[TIDZ]]]
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:   iree_vector_ext.thread_ids %[[LIN_ID]]
//       CHECK:   %[[INIT_READ0:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT0:.+]] = vector.insert_strided_slice %[[INIT_READ0]]
//       CHECK:   %[[INIT_READ1:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT1:.+]] = vector.insert_strided_slice %[[INIT_READ1]]
//       CHECK:   %[[INIT_READ2:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT2:.+]] = vector.insert_strided_slice %[[INIT_READ2]]
//       CHECK:   %[[INIT_READ3:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT3:.+]] = vector.insert_strided_slice %[[INIT_READ3]]
//       CHECK:   %[[INIT_READ4:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT4:.+]] = vector.insert_strided_slice %[[INIT_READ4]]
//       CHECK:   %[[INIT_READ5:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT5:.+]] = vector.insert_strided_slice %[[INIT_READ5]]
//       CHECK:   %[[INIT_READ6:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT6:.+]] = vector.insert_strided_slice %[[INIT_READ6]]
//       CHECK:   %[[INIT_READ7:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<1x1xf32>
//       CHECK:   %[[INIT7:.+]] = vector.insert_strided_slice %[[INIT_READ7]]
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}} = %[[INIT7]]) -> (vector<1x1x8x1x1x1xf32>)
//       CHECK:     %[[LLOAD0:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[LLOAD1:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[RLOAD0:.+]] = vector.transfer_read {{.*}} permutation_map = #[[$MAP1]]} : memref<16x256xf16, {{.*}}>, vector<8x1xf16>
//       CHECK:     %[[RLOAD1:.+]] = vector.transfer_read {{.*}} permutation_map = #[[$MAP1]]} : memref<16x256xf16, {{.*}}>, vector<8x1xf16>
//       CHECK:     vector.transfer_write %[[LLOAD0]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[LLOAD1]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD0]], %[[RHS_ALLOC]]{{.*}} : vector<8x1xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD1]], %[[RHS_ALLOC]]{{.*}} : vector<8x1xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     gpu.barrier
// CHECK-COUNT-2:   vector.transfer_read %[[LHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<1x16xf16>
// CHECK-COUNT-2:   vector.transfer_read %[[RHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<16x1xf16>
// CHECK-COUNT-2:   amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<8xf32>
//       CHECK:     %[[BCAST:.+]] = vector.broadcast {{.*}} : vector<8x1x1x1xf32> to vector<1x1x8x1x1x1xf32>
//       CHECK:     scf.yield %[[BCAST]] : vector<1x1x8x1x1x1xf32>
// CHECK-COUNT-8:  vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<1x1xf32>, memref<16x16xf32{{.*}}>
