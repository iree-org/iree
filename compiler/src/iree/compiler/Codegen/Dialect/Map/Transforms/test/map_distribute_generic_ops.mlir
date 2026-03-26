// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

#layout = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @shape_cast
// CHECK:       %[[SRC:.*]] = iree_vector_ext.to_simt %arg0 : vector<128xf16> -> vector<4xf16>
// CHECK:       %[[SC:.*]] = vector.shape_cast %[[SRC]] : vector<4xf16> to vector<1x4xf16>
// CHECK:       iree_vector_ext.to_simd %[[SC]] : vector<1x4xf16> -> vector<4x32xf16>
func.func @shape_cast(%arg0: vector<128xf16>) -> vector<4x32xf16> {
  %a = iree_vector_ext.to_layout %arg0 to layout(#layout) : vector<128xf16>
  %sc = vector.shape_cast %a : vector<128xf16> to vector<4x32xf16>
  func.return %sc : vector<4x32xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// 2D -> 1D contraction in distributed space.

#layout_sc_2d = #iree_map.pack_layout<((4), (8, 4)) : ((8), (1, 0))>

// CHECK-LABEL: @shape_cast_contract
// CHECK:       %[[SRC:.*]] = iree_vector_ext.to_simt %arg0 : vector<4x32xf16> -> vector<1x4xf16>
// CHECK:       %[[SC:.*]] = vector.shape_cast %[[SRC]] : vector<1x4xf16> to vector<4xf16>
// CHECK:       iree_vector_ext.to_simd %[[SC]] : vector<4xf16> -> vector<128xf16>
func.func @shape_cast_contract(%arg0: vector<4x32xf16>) -> vector<128xf16> {
  %a = iree_vector_ext.to_layout %arg0 to layout(#layout_sc_2d) : vector<4x32xf16>
  %sc = vector.shape_cast %a : vector<4x32xf16> to vector<128xf16>
  func.return %sc : vector<128xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_bcast_scalar = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @broadcast_scalar
// CHECK:       %[[BCAST:.*]] = vector.broadcast %arg0 : f16 to vector<4xf16>
// CHECK:       iree_vector_ext.to_simd %[[BCAST]] : vector<4xf16> -> vector<128xf16>
func.func @broadcast_scalar(%arg0: f16) -> vector<128xf16> {
  %b = vector.broadcast %arg0 : f16 to vector<128xf16>
  %bl = iree_vector_ext.to_layout %b to layout(#layout_bcast_scalar) : vector<128xf16>
  func.return %bl : vector<128xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_bcast_src = #iree_map.pack_layout<((8, 4)) : ((4, 0))>
#layout_bcast_2d = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @broadcast_vector_add_dim
// CHECK:       %[[SRC:.*]] = iree_vector_ext.to_simt %arg0 : vector<32xf16> -> vector<4xf16>
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[SRC]] : vector<4xf16> to vector<2x4xf16>
// CHECK:       iree_vector_ext.to_simd %[[BCAST]] : vector<2x4xf16> -> vector<8x32xf16>
func.func @broadcast_vector_add_dim(%arg0: vector<32xf16>) -> vector<8x32xf16> {
  %a = iree_vector_ext.to_layout %arg0 to layout(#layout_bcast_src) : vector<32xf16>
  %b = vector.broadcast %a : vector<32xf16> to vector<8x32xf16>
  %bl = iree_vector_ext.to_layout %b to layout(#layout_bcast_2d) : vector<8x32xf16>
  func.return %bl : vector<8x32xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_tr_2d = #iree_map.pack_layout<((4, 2), (8, 4)) : ((1, 0), (4, 0))>

// CHECK-LABEL: @transpose_2d
// CHECK:       %[[SRC:.*]] = iree_vector_ext.to_simt %arg0 : vector<8x32xf16> -> vector<2x4xf16>
// CHECK:       %[[T:.*]] = vector.transpose %[[SRC]], [1, 0] : vector<2x4xf16> to vector<4x2xf16>
// CHECK:       iree_vector_ext.to_simd %[[T]] : vector<4x2xf16> -> vector<32x8xf16>
func.func @transpose_2d(%arg0: vector<8x32xf16>) -> vector<32x8xf16> {
  %vl = iree_vector_ext.to_layout %arg0 to layout(#layout_tr_2d) : vector<8x32xf16>
  %t = vector.transpose %vl, [1, 0] : vector<8x32xf16> to vector<32x8xf16>
  func.return %t : vector<32x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// Dim 0 has 3 value leaves -> distributed dims [0, 1, 2].
// Dim 1 has 1 value leaf  -> distributed dim [3].
// Perm [1, 0] -> expanded [3, 0, 1, 2].

#layout_tr_complex = #iree_map.pack_layout<
  ((2, 2, 2, 2, 2, 2), (4, 8)) : ((1, 0, 2, 0, 4, 0), (8, 0))
>

// CHECK-LABEL: @transpose_multi_leaf
// CHECK:       %[[SRC:.*]] = iree_vector_ext.to_simt %arg0 : vector<64x32xf16> -> vector<2x2x2x8xf16>
// CHECK:       vector.transpose %[[SRC]], [3, 0, 1, 2] : vector<2x2x2x8xf16> to vector<8x2x2x2xf16>
func.func @transpose_multi_leaf(%arg0: vector<64x32xf16>) -> vector<32x64xf16> {
  %vl = iree_vector_ext.to_layout %arg0 to layout(#layout_tr_complex) : vector<64x32xf16>
  %t = vector.transpose %vl, [1, 0] : vector<64x32xf16> to vector<32x64xf16>
  func.return %t : vector<32x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// Thread leaf outer (32, stride=1), value leaf inner (4, stride=0).
// Value offsets: [0, 1, 2, 3]. Thread offset: (tid % 32) * 4.

#layout_step_contig = #iree_map.pack_layout<((32, 4)) : ((1, 0))>

// CHECK-LABEL: @step_contiguous
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[DELIN:.*]]:2 = affine.delinearize_index %[[TID]] into (32) : index, index
// CHECK:       %[[LIN:.*]] = affine.linearize_index disjoint [%[[DELIN]]#1, %[[C0]]] by (32, 4) : index
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[LIN]] : index to vector<4xindex>
// CHECK:       %[[ADD:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<4xindex>
// CHECK:       iree_vector_ext.to_simd %[[ADD]] : vector<4xindex> -> vector<128xindex>
func.func @step_contiguous() -> vector<128xindex> {
  %step = vector.step : vector<128xindex>
  %sl = iree_vector_ext.to_layout %step to layout(#layout_step_contig) : vector<128xindex>
  func.return %sl : vector<128xindex>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// Value leaf outer (4, stride=0), thread leaf inner (8, stride=1).
// Value offsets are strided: [0, 8, 16, 24]. Thread offset: tid % 8.

#layout_step_strided = #iree_map.pack_layout<((4, 8)) : ((0, 1))>

// CHECK-LABEL: @step_strided
// CHECK-DAG:   %[[CST:.*]] = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[DELIN:.*]]:2 = affine.delinearize_index %[[TID]] into (8) : index, index
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[DELIN]]#1 : index to vector<4xindex>
// CHECK:       %[[ADD:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<4xindex>
// CHECK:       iree_vector_ext.to_simd %[[ADD]] : vector<4xindex> -> vector<32xindex>
func.func @step_strided() -> vector<32xindex> {
  %step = vector.step : vector<32xindex>
  %sl = iree_vector_ext.to_layout %step to layout(#layout_step_strided) : vector<32xindex>
  func.return %sl : vector<32xindex>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// Two value leaves (4, 4) sandwiching a thread leaf (2).
// 2D distributed shape: vector<4x4xindex>.
// val[i][j] = i*8 + j. Thread offset: (tid % 2) * 4.

#layout_step_multi = #iree_map.pack_layout<((4, 2, 4)) : ((0, 1, 0))>

// CHECK-LABEL: @step_multi_value_leaf
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[CST:.*]] = arith.constant dense<{{\[\[}}0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19], [24, 25, 26, 27]]> : vector<4x4xindex>
// CHECK:       %[[TID:.*]] = gpu.thread_id  x
// CHECK:       %[[DELIN:.*]]:2 = affine.delinearize_index %[[TID]] into (2) : index, index
// CHECK:       %[[LIN:.*]] = affine.linearize_index disjoint [%[[DELIN]]#1, %[[C0]]] by (2, 4) : index
// CHECK:       %[[BCAST:.*]] = vector.broadcast %[[LIN]] : index to vector<4x4xindex>
// CHECK:       %[[ADD:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<4x4xindex>
// CHECK:       iree_vector_ext.to_simd %[[ADD]] : vector<4x4xindex> -> vector<32xindex>
func.func @step_multi_value_leaf() -> vector<32xindex> {
  %step = vector.step : vector<32xindex>
  %sl = iree_vector_ext.to_layout %step to layout(#layout_step_multi) : vector<32xindex>
  func.return %sl : vector<32xindex>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
