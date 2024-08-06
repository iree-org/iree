// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-cast-type-to-fit-mma))' -mlir-print-local-scope %s | FileCheck %s

func.func @mfma_matmul_96x64x16_mm(%lhs: vector<96x16xf16>, %rhs: vector<16x64xf16>, %init: vector<96x64xf16>) -> vector<96x64xf16> attributes {
    mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
      subgroup_m_count = 1, subgroup_n_count = 1>,
    workgroup_size = [64, 1, 1]} {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf16>
  return %0 : vector<96x64xf16>
}

// CHECK-LABEL: func.func @mfma_matmul_96x64x16_mm
//  CHECK-SAME: (%[[A:.+]]: vector<96x16xf16>, %[[B:.+]]: vector<16x64xf16>, %[[INIT:.+]]: vector<96x64xf16>)
//       CHECK:   %[[EXT:.+]] = arith.extf %[[INIT]] : vector<96x64xf16> to vector<96x64xf32>
//       CHECK:   %[[MM:.+]] = vector.contract
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
//  CHECK-SAME        iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
//  CHECK-SAME:     %[[A]], %[[B]], %[[EXT]] : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf32>
//       CHECK:   %[[TRUNC:.+]] = arith.truncf %[[MM]] : vector<96x64xf32> to vector<96x64xf16>
//       CHECK:   return %[[TRUNC]] : vector<96x64xf16>

// -----

func.func @mfma_matmul_96x64x16_mmt(%lhs: vector<96x16xf16>, %rhs: vector<64x16xf16>, %init: vector<96x64xf16>) -> vector<96x64xf16> attributes {
    mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
      subgroup_m_count = 1, subgroup_n_count = 1>,
    workgroup_size = [64, 1, 1]} {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<64x16xf16> into vector<96x64xf16>
  return %0 : vector<96x64xf16>
}

// CHECK-LABEL: func.func @mfma_matmul_96x64x16_mmt
//  CHECK-SAME: (%[[A:.+]]: vector<96x16xf16>, %[[B:.+]]: vector<64x16xf16>, %[[INIT:.+]]: vector<96x64xf16>)
//       CHECK:   arith.extf
//       CHECK:   vector.contract
//  CHECK-SAME:     : vector<96x16xf16>, vector<64x16xf16> into vector<96x64xf32>
//       CHECK:   arith.truncf

// -----

func.func @mfma_matmul_96x64x16_mm_cannot_downcast(%lhs: vector<96x16xf16>, %rhs: vector<16x64xf16>, %init: vector<96x64xf64>) -> vector<96x64xf64> attributes {
    mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
      subgroup_m_count = 1, subgroup_n_count = 1>,
    workgroup_size = [64, 1, 1]} {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf64>
  return %0 : vector<96x64xf64>
}

// CHECK-LABEL: func.func @mfma_matmul_96x64x16_mm_cannot_downcast
//   CHECK-NOT:   arith.extf
//       CHECK:   vector.contract
//  CHECK-SAME:     %{{.+}}, %{{.+}}, %{{.+}} : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf64>
//   CHECK-NOT:   arith.truncf

// -----

func.func @wmma_matmul_48x32x32_mm(%lhs: vector<48x32xf16>, %rhs: vector<32x32xf16>, %init: vector<48x32xf16>) -> vector<48x32xf16> attributes {
    mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>,
      subgroup_m_count = 1, subgroup_n_count = 1>,
    workgroup_size = [32, 1, 1]} {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init : vector<48x32xf16>, vector<32x32xf16> into vector<48x32xf16>
  return %0 : vector<48x32xf16>
}

// CHECK-LABEL: func.func @wmma_matmul_48x32x32_mm
//  CHECK-SAME: (%[[A:.+]]: vector<48x32xf16>, %[[B:.+]]: vector<32x32xf16>, %[[INIT:.+]]: vector<48x32xf16>)
//       CHECK:   %[[EXT:.+]] = arith.extf %[[INIT]] : vector<48x32xf16> to vector<48x32xf32>
//       CHECK:   %[[MM:.+]] = vector.contract
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
//  CHECK-SAME        iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
//  CHECK-SAME:     %[[A]], %[[B]], %[[EXT]] : vector<48x32xf16>, vector<32x32xf16> into vector<48x32xf32>
//       CHECK:   %[[TRUNC:.+]] = arith.truncf %[[MM]] : vector<48x32xf32> to vector<48x32xf16>
//       CHECK:   return %[[TRUNC]] : vector<48x32xf16>

// -----

// This tests cast_type_to_fit_mma works on IR structure coming out of transform_dialect.

// IR generated in transform_dialect is different from the one in C++ pipeline.
// it will not have mma_schedule on function attributes, but instead it will have
// "iree.amdgpu.mma" attribute directly on vector.contract.

func.func @transform_dialect_mfma_matmul_96x64x16(%lhs: vector<96x16xf16>, %rhs: vector<16x64xf16>, %init: vector<96x64xf16>) -> vector<96x64xf16> attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [64, 1, 1] subgroup_size = 64>} {
    %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
      %lhs, %rhs, %init
      {iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}
      : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf16>
  return %0 : vector<96x64xf16>
}

// CHECK-LABEL: func.func @transform_dialect_mfma_matmul_96x64x16
//  CHECK-SAME: (%[[A:.+]]: vector<96x16xf16>, %[[B:.+]]: vector<16x64xf16>, %[[INIT:.+]]: vector<96x64xf16>)
//       CHECK:   %[[EXT:.+]] = arith.extf %[[INIT]] : vector<96x64xf16> to vector<96x64xf32>
//       CHECK:   %[[MM:.+]] = vector.contract
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
//  CHECK-SAME        iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
//  CHECK-SAME:     %[[A]], %[[B]], %[[EXT]] : vector<96x16xf16>, vector<16x64xf16> into vector<96x64xf32>
//       CHECK:   %[[TRUNC:.+]] = arith.truncf %[[MM]] : vector<96x64xf32> to vector<96x64xf16>
//       CHECK:   return %[[TRUNC]] : vector<96x64xf16>
