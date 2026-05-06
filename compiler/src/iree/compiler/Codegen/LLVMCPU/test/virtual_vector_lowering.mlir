// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-virtual-vector-lowering))" --split-input-file %s | FileCheck %s

// For an n-D row-major contiguous memref, the gather lowering should add the
// gather index directly into the innermost offset rather than emit a per-lane
// affine.linearize_index/affine.delinearize_index pair.

func.func @gather_contiguous_3d(%buffer: memref<50x40x40xi8>,
                                %idx: vector<8xindex>,
                                %mask: vector<8xi1>,
                                %pass: vector<8xi8>,
                                %i: index, %j: index, %k: index)
    -> vector<8xi8> {
  %r = vector.gather %buffer[%i, %j, %k] [%idx], %mask, %pass
     : memref<50x40x40xi8>, vector<8xindex>, vector<8xi1>, vector<8xi8>
       into vector<8xi8>
  return %r : vector<8xi8>
}

// CHECK-LABEL:   func.func @gather_contiguous_3d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<50x40x40xi8>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index, %[[K:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[K]],
// CHECK:       vector.load %[[BUF]][%[[I]], %[[J]], %{{.+}}] : memref<50x40x40xi8>, vector<1xi8>

// -----

// For a non-contiguous (strided, non-row-major) memref, the gather lowering
// must keep the linearize/delinearize path because the per-dimension and
// linearized addresses no longer agree.

func.func @gather_non_contiguous_2d(
    %buffer: memref<10x20xf32, strided<[40, 1]>>,
    %idx: vector<4xindex>, %mask: vector<4xi1>, %pass: vector<4xf32>,
    %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<10x20xf32, strided<[40, 1]>>, vector<4xindex>,
       vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_non_contiguous_2d(
// CHECK-NOT:   vector.gather
// CHECK:       affine.linearize_index
// CHECK:       affine.delinearize_index

// -----

// Dynamic-shape identity-layout memref also qualifies as row-major contiguous,
// so the gather lowering should still take the contiguous path even though
// `memref::isStaticShapeAndContiguousRowMajor` returns false here.

func.func @gather_dynamic_identity_2d(%buffer: memref<?x?xf32>,
                                      %idx: vector<4xindex>,
                                      %mask: vector<4xi1>,
                                      %pass: vector<4xf32>,
                                      %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<?x?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
       into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_dynamic_identity_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<?x?xf32>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[J]],
// CHECK:       vector.load %[[BUF]][%[[I]], %{{.+}}] : memref<?x?xf32>, vector<1xf32>

// -----

// A static-shape memref with an explicit strided<[N, 1]> layout whose strides
// match the row-major contiguous sequence should also take the contiguous
// path via `isStaticShapeAndContiguousRowMajor`.

func.func @gather_strided_contiguous_2d(
    %buffer: memref<10x20xf32, strided<[20, 1]>>,
    %idx: vector<4xindex>, %mask: vector<4xi1>, %pass: vector<4xf32>,
    %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<10x20xf32, strided<[20, 1]>>, vector<4xindex>,
       vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_strided_contiguous_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<10x20xf32, strided<[20, 1]>>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[J]],
// CHECK:       vector.load %[[BUF]][%[[I]], %{{.+}}] : memref<10x20xf32, strided<[20, 1]>>, vector<1xf32>

// -----

// Tensor base falls through to the upstream pattern, which lowers each lane
// to a `tensor.extract`. The IREE-local pattern does not match (memref-only)
// and must not interfere.

func.func @gather_tensor_base_2d(%buffer: tensor<10x20xf32>,
                                 %idx: vector<4xindex>,
                                 %mask: vector<4xi1>,
                                 %pass: vector<4xf32>,
                                 %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : tensor<10x20xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
       into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_tensor_base_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: tensor<10x20xf32>
// CHECK-NOT:   vector.gather
// CHECK:       tensor.extract %[[BUF]]

// -----

// `iree_codegen.inner_tiled` lowering. The pass runs three pattern sets in
// sequence on the op (unroll non-unit iter dims to unit, drop the now-unit
// iter domain, lower the iter-free vector `inner_tiled` to
// `llvm.call_intrinsic` via the kind's `buildUnderlyingOperations`) and ends
// with `IREE::Util::eliminateHoistableConversions` to cancel the inverse
// conversion pairs the lowering patterns leave around the ACC. On a single-
// tile AVX-512 1x16x1 f32 matmul, the end result is one `llvm.fma.v16f32`
// call with no surviving `util.hoistable_conversion`s.

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
//   CHECK-NOT:   util.hoistable_conversion
//       CHECK:   llvm.call_intrinsic "llvm.fma.v16f32"({{.*}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
