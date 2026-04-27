// RUN: iree-opt %s -split-input-file | FileCheck %s

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]

// MMA_X86_AVX512_1x8x1_F64_F64, intrinsics 1x1x1
func.func @cpu_avx512_1x8x1_f64(
    %lhs: vector<1x1x1xf64>, %rhs: vector<1x1x8xf64>, %acc: vector<1x1x8xf64>)
    -> vector<1x1x8xf64> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x8x1_F64_F64>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1xf64>, vector<1x1x8xf64> into vector<1x1x8xf64>
  return %0 : vector<1x1x8xf64>
}
// CHECK-LABEL: func @cpu_avx512_1x8x1_f64
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x8x1_F64_F64>
//  CHECK-SAME:       semantics = #iree_cpu.mma_semantics<>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512_1x16x1_F32_F32, intrinsics 2x1x1
func.func @cpu_avx512_1x16x1_f32(
    %lhs: vector<2x1x2x1xf32>, %rhs: vector<1x1x1x16xf32>, %acc: vector<2x1x2x16xf32>)
    -> vector<2x1x2x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32, intrinsics_m = 2>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<2x1x2x1xf32>, vector<1x1x1x16xf32> into vector<2x1x2x16xf32>
  return %0 : vector<2x1x2x16xf32>
}
// CHECK-LABEL: func @cpu_avx512_1x16x1_f32
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32, intrinsics_m = 2>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512_1x16x1_F32_F16_CASTF32, intrinsics 1x2x1 -> M=1,N=2,K=1
func.func @cpu_avx512_1x16x1_f16_castf32(
    %lhs: vector<1x1x1xf16>, %rhs: vector<1x2x32xf16>, %acc: vector<1x2x32xf32>)
    -> vector<1x2x32xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F16_CASTF32, intrinsics_n = 2>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1xf16>, vector<1x2x32xf16> into vector<1x2x32xf32>
  return %0 : vector<1x2x32xf32>
}
// CHECK-LABEL: func @cpu_avx512_1x16x1_f16_castf32
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F16_CASTF32, intrinsics_n = 2>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512FP16_1x32x1_F16_F16, intrinsics 1x1x2
func.func @cpu_avx512fp16_1x32x1_f16(
    %lhs: vector<1x2x1x2xf16>, %rhs: vector<2x1x32x2xf16>, %acc: vector<1x1x1x32xf16>)
    -> vector<1x1x1x32xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512FP16_1x32x1_F16_F16, intrinsics_k = 2>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x2x1x2xf16>, vector<2x1x32x2xf16> into vector<1x1x1x32xf16>
  return %0 : vector<1x1x1x32xf16>
}
// CHECK-LABEL: func @cpu_avx512fp16_1x32x1_f16
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512FP16_1x32x1_F16_F16, intrinsics_k = 2>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512BF16_1x16x2_F32_BF16, intrinsics 2x2x1
func.func @cpu_avx512bf16_1x16x2_bf16(
    %lhs: vector<2x1x2x2xbf16>, %rhs: vector<1x2x32x2xbf16>, %acc: vector<2x2x2x32xf32>)
    -> vector<2x2x2x32xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512BF16_1x16x2_F32_BF16, intrinsics_m = 2, intrinsics_n = 2>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<2x1x2x2xbf16>, vector<1x2x32x2xbf16> into vector<2x2x2x32xf32>
  return %0 : vector<2x2x2x32xf32>
}
// CHECK-LABEL: func @cpu_avx512bf16_1x16x2_bf16
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512BF16_1x16x2_F32_BF16, intrinsics_m = 2, intrinsics_n = 2>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512_1x16x2_I32_I16, intrinsics 1x4x1
func.func @cpu_avx512_1x16x2_i32_i16(
    %lhs: vector<1x1x1x2xi16>, %rhs: vector<1x4x64x2xi16>, %acc: vector<1x4x1x64xi32>)
    -> vector<1x4x1x64xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x2_I32_I16, intrinsics_n = 4>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1x2xi16>, vector<1x4x64x2xi16> into vector<1x4x1x64xi32>
  return %0 : vector<1x4x1x64xi32>
}
// CHECK-LABEL: func @cpu_avx512_1x16x2_i32_i16
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x2_I32_I16, intrinsics_n = 4>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512VNNI_1x16x2_I32_I16, intrinsics 2x1x2
func.func @cpu_avx512vnni_1x16x2_i32_i16(
    %lhs: vector<2x2x2x4xi16>, %rhs: vector<2x1x16x4xi16>, %acc: vector<2x1x2x16xi32>)
    -> vector<2x1x2x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512VNNI_1x16x2_I32_I16, intrinsics_m = 2, intrinsics_k = 2>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<2x2x2x4xi16>, vector<2x1x16x4xi16> into vector<2x1x2x16xi32>
  return %0 : vector<2x1x2x16xi32>
}
// CHECK-LABEL: func @cpu_avx512vnni_1x16x2_i32_i16
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512VNNI_1x16x2_I32_I16, intrinsics_m = 2, intrinsics_k = 2>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512_1x16x2_I32_I8_CASTI16, intrinsics 1x1x4
func.func @cpu_avx512_1x16x2_i32_i8(
    %lhs: vector<1x4x1x8xi8>, %rhs: vector<4x1x16x8xi8>, %acc: vector<1x1x1x16xi32>)
    -> vector<1x1x1x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x2_I32_I8_CASTI16, intrinsics_k = 4>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x4x1x8xi8>, vector<4x1x16x8xi8> into vector<1x1x1x16xi32>
  return %0 : vector<1x1x1x16xi32>
}
// CHECK-LABEL: func @cpu_avx512_1x16x2_i32_i8
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x2_I32_I8_CASTI16, intrinsics_k = 4>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16, intrinsics 4x1x2
func.func @cpu_avx512vnni_1x16x2_i32_i8(
    %lhs: vector<4x2x4x4xi8>, %rhs: vector<2x1x16x4xi8>, %acc: vector<4x1x4x16xi32>)
    -> vector<4x1x4x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16, intrinsics_m = 4, intrinsics_k = 2>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<4x2x4x4xi8>, vector<2x1x16x4xi8> into vector<4x1x4x16xi32>
  return %0 : vector<4x1x4x16xi32>
}
// CHECK-LABEL: func @cpu_avx512vnni_1x16x2_i32_i8
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16, intrinsics_m = 4, intrinsics_k = 2>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
func.func @cpu_inner_tiled_tensor_avx512_f32_packed_m16(
    %lhs: tensor<2x4x16x1xf32>, %rhs: tensor<4x3x16x1xf32>, %acc: tensor<2x3x16x16xf32>)
    -> tensor<2x3x16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32, intrinsics_m = 16>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<2x4x16x1xf32>, tensor<4x3x16x1xf32> into tensor<2x3x16x16xf32>
  return %0 : tensor<2x3x16x16xf32>
}
// CHECK-LABEL: func @cpu_inner_tiled_tensor_avx512_f32_packed_m16
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32, intrinsics_m = 16>
//  CHECK-SAME:       semantics = #iree_cpu.mma_semantics<>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// Same as the f32 testcase above, but using bf16 so the undistributed LHS/RHS
// tiles are 16×2 instead of 16×1. This exercises the non-unit RHS N/K shape
// mismatch that would occur without using the physical N×K MMA tile layout.
func.func @cpu_inner_tiled_tensor_avx512bf16_packed_m16(
    %lhs: tensor<2x4x16x2xbf16>, %rhs: tensor<4x3x16x2xbf16>, %acc: tensor<2x3x16x16xf32>)
    -> tensor<2x3x16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512BF16_1x16x2_F32_BF16, intrinsics_m = 16>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<2x4x16x2xbf16>, tensor<4x3x16x2xbf16> into tensor<2x3x16x16xf32>
  return %0 : tensor<2x3x16x16xf32>
}
// CHECK-LABEL: func @cpu_inner_tiled_tensor_avx512bf16_packed_m16
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512BF16_1x16x2_F32_BF16, intrinsics_m = 16>
//  CHECK-SAME:       semantics = #iree_cpu.mma_semantics<>

// -----

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
// AArch64 SVE `fmla`: scalable N in the RHS/acc MMA tiles (`vector<1x[4]xf32>`)
// pairs with dynamic (`?`) tensor inner extents on N.
func.func @cpu_inner_tiled_tensor_arm_sve_fmla_f32(
    %lhs: tensor<2x2x1x1xf32>, %rhs: tensor<2x2x1x?xf32>, %acc: tensor<2x2x1x?xf32>)
    -> tensor<2x2x1x?xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<2x2x1x1xf32>, tensor<2x2x1x?xf32> into tensor<2x2x1x?xf32>
  return %0 : tensor<2x2x1x?xf32>
}
// CHECK-LABEL: func @cpu_inner_tiled_tensor_arm_sve_fmla_f32
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32>
//  CHECK-SAME:       semantics = #iree_cpu.mma_semantics<>
