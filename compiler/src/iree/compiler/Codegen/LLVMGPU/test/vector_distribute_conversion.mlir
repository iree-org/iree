// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmgpu-vector-distribute, canonicalize, cse))' -split-input-file %s | FileCheck %s

func.func @matmul_256x256x256(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                              %rhs: memref<256x16xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                              %out: memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
  attributes {
    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
                     subgroup_m_count = 1, subgroup_n_count = 1, subgroup_m_tile_count = 1, subgroup_n_tile_count = 1, subgroup_k_tile_count = 2>,
    subgroup_size = 64, workgroup_size = [64, 1, 1]} {
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

// CHECK-LABEL: func.func @matmul_256x256x256
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x4xf32>
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}} = %[[INIT]]) -> (vector<1x1x1x1x1x4xf32>)
//       CHECK:     %[[LLOAD:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[RLOAD:.+]] = vector.transfer_read {{.*}} : memref<256x16xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     vector.transfer_write %[[LLOAD]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD]], %[[RHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     gpu.barrier
// CHECK-COUNT-2:   vector.transfer_read %[[LHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<1x4xf16>
// CHECK-COUNT-2:   vector.transfer_read %[[RHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
// CHECK-COUNT-2:   amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:     %[[BCAST:.+]] = vector.broadcast {{.*}} : vector<1x1x1x4xf32> to vector<1x1x1x1x1x4xf32>
//       CHECK:     scf.yield %[[BCAST]] : vector<1x1x1x1x1x4xf32>
//       CHECK:  vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<16x16xf32{{.*}}>

// -----

func.func @matmul_256x256x256(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                              %rhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                              %out: memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
  attributes {
    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
                     subgroup_m_count = 1, subgroup_n_count = 1, subgroup_m_tile_count = 1, subgroup_n_tile_count = 1, subgroup_k_tile_count = 2>,
    subgroup_size = 64, workgroup_size = [64, 1, 1]} {
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

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @matmul_256x256x256
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[INIT_READ:.+]] = vector.transfer_read %{{.*}} memref<16x16xf32, {{.*}}>, vector<4x1xf32>
//       CHECK:   %[[INIT_TRANSP:.+]] = vector.transpose %[[INIT_READ]], [1, 0]
//       CHECK:   %[[INIT:.+]] = vector.insert_strided_slice %[[INIT_TRANSP]]
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}} = %[[INIT]]) -> (vector<1x1x1x1x1x4xf32>)
//       CHECK:     %[[LLOAD:.+]] = vector.transfer_read {{.*}} : memref<16x256xf16, {{.*}}>, vector<1x8xf16>
//       CHECK:     %[[RLOAD:.+]] = vector.transfer_read {{.*}} permutation_map = #[[$MAP]]} : memref<16x256xf16, {{.*}}>, vector<8x1xf16>
//       CHECK:     vector.transfer_write %[[LLOAD]], %[[LHS_ALLOC]]{{.*}} : vector<1x8xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %[[RLOAD]], %[[RHS_ALLOC]]{{.*}} : vector<8x1xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:     gpu.barrier
// CHECK-COUNT-2:   vector.transfer_read %[[LHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<1x4xf16>
// CHECK-COUNT-2:   vector.transfer_read %[[RHS_ALLOC]][{{.+}}], %{{.+}} {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
// CHECK-COUNT-2:   amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:     %[[BCAST:.+]] = vector.broadcast {{.*}} : vector<1x1x1x4xf32> to vector<1x1x1x1x1x4xf32>
//       CHECK:     scf.yield %[[BCAST]] : vector<1x1x1x1x1x4xf32>
//       CHECK:  vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<16x16xf32{{.*}}>
