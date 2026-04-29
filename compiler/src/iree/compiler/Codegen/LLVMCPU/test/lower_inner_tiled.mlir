// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-llvmcpu-lower-inner-tiled))" --split-input-file | FileCheck %s

// `iree-llvmcpu-lower-inner-tiled` runs two pattern sets in sequence on a
// vector-semantics `iree_codegen.inner_tiled` (the kind produced by
// GenericVectorization via the VectorizableOpInterface external model):
//   1. drop the (now-unit) iter domain,
//   2. lower the iter-free vector inner_tiled to llvm.call_intrinsic.
// On a single-tile AVX-512 1x16x1 f32 matmul the end result is one
// `llvm.fma.v16f32` call.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @lower_avx512_1x16x1_f32(
    %lhs: vector<1x1x1x1xf32>, %rhs: vector<1x1x16x1xf32>,
    %acc: vector<1x1x1x16xf32>) -> vector<1x1x1x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1x1xf32>, vector<1x1x16x1xf32> into vector<1x1x1x16xf32>
  return %0 : vector<1x1x1x16xf32>
}

// CHECK-LABEL: func @lower_avx512_1x16x1_f32
//   CHECK-NOT:   iree_codegen.inner_tiled
//       CHECK:   llvm.call_intrinsic "llvm.fma.v16f32"({{.*}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
