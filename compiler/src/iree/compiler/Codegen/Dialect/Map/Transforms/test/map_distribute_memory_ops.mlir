// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

// 1D gather with a non-coalescable depth-3 layout and a mask.
// Layout (2, 3, 4, 2):(12, 0, 3, 0) has interleaved thread/value leaves:
//   Thread: (2, stride=12), (4, stride=3) → 8 threads
//   Value:  (3, dataStride=8), (2, dataStride=1) → distributed shape [3, 2]
// The two value leaves don't coalesce (3*1 ≠ 8).
// After distribution: the step resolves to dense<[[0,1],[8,9],[16,17]]>,
// the mask is distributed, and the thread offset is absorbed into the base.

#layout_deep = #iree_map.pack_layout<((2, 3, 4, 2)) : ((12, 0, 3, 0))>

// CHECK-LABEL: @transfer_gather_deep_masked
// CHECK-DAG:   %[[OFFSETS:.*]] = arith.constant dense<{{\[\[}}0, 1], [8, 9], [16, 17]]> : vector<3x2xindex>
// CHECK:       %[[MASK:.*]] = iree_vector_ext.to_simt %arg1 : vector<48xi1> -> vector<3x2xi1>
// CHECK:       %[[GATHER:.*]] = iree_vector_ext.transfer_gather
// CHECK-SAME:    [%[[OFFSETS]] : vector<3x2xindex>]
// CHECK-SAME:    %[[MASK]]
// CHECK-SAME:    memref<?xf16>, vector<3x2xf16>, vector<3x2xi1>
// CHECK:       iree_vector_ext.to_simd %[[GATHER]] : vector<3x2xf16> -> vector<48xf16>
func.func @transfer_gather_deep_masked(%mem: memref<?xf16>, %mask: vector<48xi1>) -> vector<48xf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f16
  %mask_l = iree_vector_ext.to_layout %mask to layout(#layout_deep) : vector<48xi1>
  %v = iree_vector_ext.transfer_gather %mem[%c0], %pad, %mask_l {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>]
  } : memref<?xf16>, vector<48xf16>, vector<48xi1>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_deep) : vector<48xf16>
  func.return %vl : vector<48xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// 2D gather: outer dim gathered via index vec, inner dim contiguous.
// Layout: mode 0 = (4, 2):(1, 0), mode 1 = (8, 8):(8, 0) → 32 threads.
// Distributed shape: [2, 8].
// After distribution: the index vec is distributed via to_simt, the inner
// contiguous dim becomes a step dense<[0,1,...,7]>.

#layout_2d = #iree_map.pack_layout<((4, 2), (8, 8)) : ((1, 0), (8, 0))>
#layout_idx = #iree_map.pack_layout<((4, 2)) : ((1, 0))>

// CHECK-LABEL: @transfer_gather_indexed_outer
// CHECK-DAG:   %[[INNER:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
// CHECK:       %[[IDX:.*]] = iree_vector_ext.to_simt %arg1 : vector<8xindex> -> vector<2xindex>
// CHECK:       %[[GATHER:.*]] = iree_vector_ext.transfer_gather
// CHECK-SAME:    [%[[IDX]], %[[INNER]] : vector<2xindex>, vector<8xindex>]
// CHECK-SAME:    memref<?x64xf16>, vector<2x8xf16>
// CHECK:       iree_vector_ext.to_simd %[[GATHER]] : vector<2x8xf16> -> vector<8x64xf16>
func.func @transfer_gather_indexed_outer(%mem: memref<?x64xf16>, %idx: vector<8xindex>) -> vector<8x64xf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f16
  %idx_l = iree_vector_ext.to_layout %idx to layout(#layout_idx) : vector<8xindex>
  %v = iree_vector_ext.transfer_gather %mem[%c0, %c0]
      [%idx_l : vector<8xindex>], %pad {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>]
  } : memref<?x64xf16>, vector<8x64xf16>
  %vl = iree_vector_ext.to_layout %v to layout(#layout_2d) : vector<8x64xf16>
  func.return %vl : vector<8x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
