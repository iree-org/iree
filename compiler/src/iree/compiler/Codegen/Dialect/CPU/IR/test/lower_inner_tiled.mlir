// RUN: iree-opt %s -iree-transform-dialect-interpreter -iree-transform-dialect-drop-schedule --split-input-file | FileCheck %s

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
//       CHECK:   %[[SCALAR:.+]] = vector.extract {{.*}} : f32 from vector<{{1x1|1}}xf32>
//       CHECK:   %[[BCST:.+]] = vector.broadcast %[[SCALAR]] : f32 to vector<16xf32>
//       CHECK:   llvm.call_intrinsic "llvm.fma.v16f32"(%{{.+}}, %[[BCST]], %{{.+}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>

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
//       CHECK:   arith.extf {{.*}} : vector<1xf16> to vector<1xf32>
//       CHECK:   arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
//       CHECK:   %[[SCALAR:.+]] = vector.extract {{.*}} : f32 from vector<1xf32>
//       CHECK:   %[[BCST:.+]] = vector.broadcast %[[SCALAR]] : f32 to vector<16xf32>
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
//       CHECK:   arith.extsi {{.*}} : vector<2xi8> to vector<2xi16>
//       CHECK:   arith.extsi {{.*}} : vector<32xi8> to vector<32xi16>
//       CHECK:   %[[SCALAR:.+]] = vector.extract {{.*}} : i32 from vector<1xi32>
//       CHECK:   %[[BCST:.+]] = vector.broadcast %[[SCALAR]] : i32 to vector<16xi32>
//       CHECK:   vector.bitcast %[[BCST]] : vector<16xi32> to vector<32xi16>
//       CHECK:   %[[DOT:.+]] = llvm.call_intrinsic "llvm.x86.avx512.pmaddw.d.512"({{.*}}) : (vector<32xi16>, vector<32xi16>) -> vector<16xi32>
//       CHECK:   arith.addi {{.*}}, %[[DOT]] : vector<16xi32>

// -----

// The MMA_GENERIC_SCALAR_1x1x1_REG* family is type-polymorphic and 1×1×1;
// after applying `intrinsics_m` / `intrinsics_n` / `intrinsics_k`, the
// operand tiles are row-major (M, K) / (N, K) / (M, N) — exactly the shape
// `linalg.mmt4d` vectorizes to. So these lower directly to a single
// `vector.contract` over the unrolled tile, bypassing the swizzle-
// distribute machinery the other (architecture-specific) intrinsics use.
// Mixed-precision element types (e.g. bf16 inputs, f32 accumulator) are
// handled by an explicit `arith.extf` widen of LHS/RHS to ACC's element
// type before the contract; a homogeneous integer case lowers analogously
// through `arith.extsi`. The `_REG*` budget suffix is a property of the
// cost model — at lowering time all variants behave identically.

#contraction_accesses_g = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_generic_bf16_f32(
    %lhs: vector<8x1xbf16>, %rhs: vector<8x1xbf16>, %acc: vector<8x8xf32>)
    -> vector<8x8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_g,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_GENERIC_SCALAR_1x1x1_REG8, intrinsics_m = 8, intrinsics_n = 8, lhs_type = bf16, rhs_type = bf16, acc_type = f32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<8x1xbf16>, vector<8x1xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

func.func @lower_generic_i32_i8(
    %lhs: vector<4x1xi8>, %rhs: vector<4x1xi8>, %acc: vector<4x4xi32>)
    -> vector<4x4xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_g,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_GENERIC_SCALAR_1x1x1_REG8, intrinsics_m = 4, intrinsics_n = 4, lhs_type = i8, rhs_type = i8, acc_type = i32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<4x1xi8>, vector<4x1xi8> into vector<4x4xi32>
  return %0 : vector<4x4xi32>
}

// Same shape as above but with unsigned LHS/RHS — widening to acc must
// use `arith.extui`, not `arith.extsi`. Storage stays signless (the inner
// tile types come from `getABCElementTypes`, which strips signedness);
// the unsigned annotation lives on the attr's `lhs_type` / `rhs_type`.
// Acc stays signed.
func.func @lower_generic_i32_ui8(
    %lhs: vector<4x1xi8>, %rhs: vector<4x1xi8>, %acc: vector<4x4xi32>)
    -> vector<4x4xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_g,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_GENERIC_SCALAR_1x1x1_REG8, intrinsics_m = 4, intrinsics_n = 4, lhs_type = ui8, rhs_type = ui8, acc_type = i32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<4x1xi8>, vector<4x1xi8> into vector<4x4xi32>
  return %0 : vector<4x4xi32>
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

// CHECK-LABEL: func @lower_generic_bf16_f32
//       CHECK:   %[[LHS_F32:.+]] = arith.extf %{{.+}} : vector<8x1xbf16> to vector<8x1xf32>
//       CHECK:   %[[RHS_F32:.+]] = arith.extf %{{.+}} : vector<8x1xbf16> to vector<8x1xf32>
//       CHECK:   vector.contract
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     kind = #vector.kind<add>
//  CHECK-SAME:     %[[LHS_F32]], %[[RHS_F32]], %{{.+}} : vector<8x1xf32>, vector<8x1xf32> into vector<8x8xf32>

// CHECK-LABEL: func @lower_generic_i32_i8
//       CHECK:   %[[LHS_I32:.+]] = arith.extsi %{{.+}} : vector<4x1xi8> to vector<4x1xi32>
//       CHECK:   %[[RHS_I32:.+]] = arith.extsi %{{.+}} : vector<4x1xi8> to vector<4x1xi32>
//       CHECK:   vector.contract
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     kind = #vector.kind<add>
//  CHECK-SAME:     %[[LHS_I32]], %[[RHS_I32]], %{{.+}} : vector<4x1xi32>, vector<4x1xi32> into vector<4x4xi32>

// CHECK-LABEL: func @lower_generic_i32_ui8
//       CHECK:   %[[LHS_UI32:.+]] = arith.extui %{{.+}} : vector<4x1xi8> to vector<4x1xi32>
//       CHECK:   %[[RHS_UI32:.+]] = arith.extui %{{.+}} : vector<4x1xi8> to vector<4x1xi32>
//       CHECK:   vector.contract
//  CHECK-SAME:     %[[LHS_UI32]], %[[RHS_UI32]], %{{.+}} : vector<4x1xi32>, vector<4x1xi32> into vector<4x4xi32>

// -----

// Generic-scalar with `intrinsics_n = 1` (narrow N): the ACC operand
// arrives as rank-3 `vector<4x1x1xf32>` rather than the rank-2 `(M, N)`
// form `vector.contract` consumes. The non-unit M dim is at position 0,
// and the trailing two unit dims come from the swizzle's `Internal(1)`
// placeholders for the K-side and N-side of the 1×1×1 base intrinsic.
// The lowering must `shape_cast` the rank-3 operand into the rank-2
// `(M, N)` shape for the contract, then `shape_cast` the contract result
// back to the original rank-3 ACC type so the inner_tiled op's result
// type is preserved downstream.

#contraction_accesses_n = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_generic_narrow_n_f32(
    %lhs: vector<4x1xf32>, %rhs: vector<1x1xf32>, %acc: vector<4x1x1xf32>)
    -> vector<4x1x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_n,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_GENERIC_SCALAR_1x1x1_REG16, intrinsics_m = 4, lhs_type = f32, rhs_type = f32, acc_type = f32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<4x1xf32>, vector<1x1xf32> into vector<4x1x1xf32>
  return %0 : vector<4x1x1xf32>
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

// CHECK-LABEL: func @lower_generic_narrow_n_f32
//       CHECK:   %[[ACC_2D:.+]] = vector.shape_cast %{{.+}} : vector<4x1x1xf32> to vector<4x1xf32>
//       CHECK:   %[[RES_2D:.+]] = vector.contract
//  CHECK-SAME:     %{{.+}}, %{{.+}}, %[[ACC_2D]] : vector<4x1xf32>, vector<1x1xf32> into vector<4x1xf32>
//       CHECK:   vector.shape_cast %[[RES_2D]] : vector<4x1xf32> to vector<4x1x1xf32>

// -----

// AVX-512 16×1×1 f32 (M↔N-swapped orientation of 1×16×1) with intrinsics_m=2,
// intrinsics_n=4. The narrow side is RHS, so broadcast widens RHS 1→16; FMA
// is LHS/RHS-symmetric so the call args don't swap.
//
// First test to exercise a non-identity ACC swizzle permutation. The
// swizzle's `expandShape` dims `[2, 16, 4, 1]` are reordered by the
// permutation `[0, 2, 1, 3]` (cross-intrinsic dims placed before internal
// dims), giving the distributed N-D shape `<2x4x16x1>`. The `inner_tiled`
// op uses a more-compact form with the trailing 1 dropped (`<2x4x16>` —
// matchable against the full shape by `matchTileTypes`' filter); the
// lowering shape_casts that into and out of the `<2x4x16x1>` form for
// the per-intrinsic distribute/reassemble.

#contraction_accesses_t = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_avx512_16x1x1_f32(
    %lhs: vector<2x16xf32>, %rhs: vector<4x1xf32>, %acc: vector<2x4x16xf32>)
    -> vector<2x4x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_t,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_16x1x1_F32_F32, intrinsics_m = 2, intrinsics_n = 4>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<2x16xf32>, vector<4x1xf32> into vector<2x4x16xf32>
  return %0 : vector<2x4x16xf32>
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

// CHECK-LABEL: func @lower_avx512_16x1x1_f32
//       CHECK:   util.hoistable_conversion "shape_cast_to_intrinsic"
//       CHECK:     vector.shape_cast {{.*}} : vector<2x4x16xf32> to vector<2x4x16x1xf32>
//       CHECK:   vector.broadcast {{.*}} : f32 to vector<16xf32>
//       CHECK:   llvm.call_intrinsic "llvm.fma.v16f32"
//       CHECK:   util.hoistable_conversion "shape_cast_from_intrinsic"
//       CHECK:     vector.shape_cast {{.*}} : vector<2x4x16x1xf32> to vector<2x4x16xf32>

// -----

// vpdpbusd is asymmetric (first byte source unsigned, second signed). The
// natural variant lowers as `(acc, lhs, rhs)`; the swapped variant flips
// the args at the call so the unsigned operand still lands first.

#contraction_accesses_b = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_avx512vnni_1x16x4_vpdpbusd(
    %lhs: vector<1x4xi8>, %rhs: vector<16x4xi8>, %acc: vector<1x16xi32>)
    -> vector<1x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_b,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512VNNI_1x16x4_I32_UI8_I8>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x4xi8>, vector<16x4xi8> into vector<1x16xi32>
  return %0 : vector<1x16xi32>
}

func.func @lower_avx512vnni_16x1x4_vpdpbusd(
    %lhs: vector<16x4xi8>, %rhs: vector<1x4xi8>, %acc: vector<16x1xi32>)
    -> vector<16x1xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_b,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512VNNI_16x1x4_I32_I8_UI8>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<16x4xi8>, vector<1x4xi8> into vector<16x1xi32>
  return %0 : vector<16x1xi32>
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

// Natural: LHS (ui8, narrow) is broadcast and routed into vpdpbusd's
// first (unsigned) byte-source slot.
// CHECK-LABEL: func @lower_avx512vnni_1x16x4_vpdpbusd(
//  CHECK-SAME:   %[[N_LHS:[a-zA-Z0-9]+]]: vector<1x4xi8>
//  CHECK-SAME:   %[[N_RHS:[a-zA-Z0-9]+]]: vector<16x4xi8>
//       CHECK:   %[[N_RHS_FLAT:.+]] = vector.shape_cast %[[N_RHS]] : vector<16x4xi8> to vector<64xi8>
//       CHECK:   %[[N_BCAST_I32:.+]] = vector.broadcast %{{.+}} : i32 to vector<16xi32>
//       CHECK:   %[[N_LHS_FLAT:.+]] = vector.bitcast %[[N_BCAST_I32]] : vector<16xi32> to vector<64xi8>
//       CHECK:   llvm.call_intrinsic "llvm.x86.avx512.vpdpbusd.512"(%{{.+}}, %[[N_LHS_FLAT]], %[[N_RHS_FLAT]])

// Swapped: RHS (ui8, narrow) is broadcast; the lowering swaps arg order at
// the call so the broadcast result still lands in vpdpbusd's first slot.
// CHECK-LABEL: func @lower_avx512vnni_16x1x4_vpdpbusd(
//  CHECK-SAME:   %[[S_LHS:[a-zA-Z0-9]+]]: vector<16x4xi8>
//  CHECK-SAME:   %[[S_RHS:[a-zA-Z0-9]+]]: vector<1x4xi8>
//       CHECK:   %[[S_LHS_FLAT:.+]] = vector.shape_cast %[[S_LHS]] : vector<16x4xi8> to vector<64xi8>
//       CHECK:   %[[S_BCAST_I32:.+]] = vector.broadcast %{{.+}} : i32 to vector<16xi32>
//       CHECK:   %[[S_RHS_FLAT:.+]] = vector.bitcast %[[S_BCAST_I32]] : vector<16xi32> to vector<64xi8>
//       CHECK:   llvm.call_intrinsic "llvm.x86.avx512.vpdpbusd.512"(%{{.+}}, %[[S_RHS_FLAT]], %[[S_LHS_FLAT]])

// -----

// AVX-512 VNNI 16×16×2 i8 → i32. Unlike the 1×16×2 / 16×1×2 CASTI16 variants
// (one row per intrinsic, with a per-row i8→i16 widen + broadcast), this
// processes whole 16×2 i8 panels: one `vpmovsxbw` widen per panel, 4 in-lane
// `vector.shuffle`s (→ `vpshufd`) fanning the LHS rows, 4 128-bit-block
// `vector.shuffle`s (→ `vbroadcasti32x4`) fanning the RHS columns, then 16
// `vpdpwssd` over the 4×4 grid. The ACC tile is the block-interleaved
// `vector<4x4x4x4xi32>` from `getIntrinsicSwizzle`.

#contraction_accesses_16x16 = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_avx512vnni_16x16x2_i8(
    %lhs: vector<16x2xi8>, %rhs: vector<16x2xi8>, %acc: vector<4x4x4x4xi32>)
    -> vector<4x4x4x4xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses_16x16,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512VNNI_16x16x2_I32_I8_CASTI16>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<16x2xi8>, vector<16x2xi8> into vector<4x4x4x4xi32>
  return %0 : vector<4x4x4x4xi32>
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

// CHECK-LABEL: func @lower_avx512vnni_16x16x2_i8
//       CHECK:   arith.extsi {{.*}} : vector<32xi8> to vector<32xi16>
//       CHECK:   arith.extsi {{.*}} : vector<32xi8> to vector<32xi16>
//   CHECK-DAG:   vector.shuffle %{{.+}} [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
//   CHECK-DAG:   vector.shuffle %{{.+}} [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
//       CHECK-COUNT-16:   llvm.call_intrinsic "llvm.x86.avx512.vpdpwssd.512"
