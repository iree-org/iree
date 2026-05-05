// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// AVX-512 1×16×1 f32 → f32 with intrinsics_m=1, intrinsics_n=1, intrinsics_k=1.
// Exercises the FMA shape (no widening) on the simplest singly-unrolled case.

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_avx512_1x16x1_f32(
    %lhs: vector<1x1xf32>, %rhs: vector<16x1xf32>, %acc: vector<1x16xf32>)
    -> vector<1x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1xf32>, vector<16x1xf32> into vector<1x16xf32>
  return %0 : vector<1x16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_avx512_1x16x1_f32
//       CHECK:   vector.broadcast {{.*}} : vector<{{1x1|1}}xf32> to vector<16x1xf32>
//       CHECK:   vector.shape_cast {{.*}} : vector<16x1xf32> to vector<16xf32>
//       CHECK:   llvm.call_intrinsic "llvm.fma.v16f32"({{.*}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>

// -----

// AVX-512 1×16×1 f16 → f32 (CASTF32). Exercises the FMA shape with arith.extf
// widening (f16 → f32) on lhs/rhs after the broadcast, before the FMA call.

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_avx512_1x16x1_f16_castf32(
    %lhs: vector<1x1xf16>, %rhs: vector<16x1xf16>, %acc: vector<1x16xf32>)
    -> vector<1x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F16_CASTF32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1xf16>, vector<16x1xf16> into vector<1x16xf32>
  return %0 : vector<1x16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_avx512_1x16x1_f16_castf32
//       CHECK:   vector.broadcast {{.*}} : vector<{{1x1|1}}xf16> to vector<16x1xf16>
//       CHECK:   vector.shape_cast {{.*}} : vector<16x1xf16> to vector<16xf16>
//       CHECK:   arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
//       CHECK:   arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
//       CHECK:   llvm.call_intrinsic "llvm.fma.v16f32"({{.*}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>

// -----

// AVX-512 (non-VNNI) 1×16×2 i8 → i32 via i16 (CASTI16). Exercises the
// DotNonAccumulating shape: arith.extsi widens i8→i16, `pmaddw.d.512`
// returns a partial dot, and the lowering finishes with arith.addi to fold
// into the accumulator.

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_avx512_1x16x2_i8_casti16(
    %lhs: vector<1x2xi8>, %rhs: vector<16x2xi8>, %acc: vector<1x16xi32>)
    -> vector<1x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x2_I32_I8_CASTI16>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x2xi8>, vector<16x2xi8> into vector<1x16xi32>
  return %0 : vector<1x16xi32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_avx512_1x16x2_i8_casti16
//       CHECK:   vector.broadcast {{.*}} : vector<{{1x2|2}}xi8> to vector<16x2xi8>
//       CHECK:   vector.shape_cast {{.*}} : vector<16x2xi8> to vector<32xi8>
//       CHECK:   arith.extsi {{.*}} : vector<32xi8> to vector<32xi16>
//       CHECK:   arith.extsi {{.*}} : vector<32xi8> to vector<32xi16>
//       CHECK:   %[[DOT:.+]] = llvm.call_intrinsic "llvm.x86.avx512.pmaddw.d.512"({{.*}}) : (vector<32xi16>, vector<32xi16>) -> vector<16xi32>
//       CHECK:   arith.addi {{.*}}, %[[DOT]] : vector<16xi32>
