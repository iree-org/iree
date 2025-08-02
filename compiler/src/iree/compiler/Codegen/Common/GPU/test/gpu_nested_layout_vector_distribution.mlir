// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 2],
  outer_tile        = [1, 1],
  thread_tile       = [4, 8],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

// CHECK-LABEL: @distribute_transfer_read_col_major
func.func @distribute_transfer_read_col_major(%arg0: memref<32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]}
                  : memref<32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_col_major) : vector<16x16xf16>
  func.return %rootl : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[YX:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 8)
// CHECK: %[[Y_SCALED:.+]] = affine.linearize_index disjoint [%[[YX]]#1, %c0] by (4, 4)
// CHECK: %[[RD00:.+]] = vector.transfer_read %arg0[%[[Y_SCALED]], %[[YX]]#2], {{.*}} : memref<32x32xf16>, vector<4x1xf16>
// CHECK: vector.insert_strided_slice %[[RD00]], %{{.*}} {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf16> into vector<1x2x1x1x4x1xf16>
// CHECK: %[[X_PLUS_BATCH:.+]] = affine.linearize_index disjoint [%c1, %[[YX]]#2] by (2, 8)
// CHECK: vector.transfer_read %arg0[%[[Y_SCALED]], %[[X_PLUS_BATCH]]], %{{.*}} {in_bounds = [true, true]} : memref<32x32xf16>, vector<4x1xf16>
// CHECK: vector.insert_strided_slice {{.*}} {offsets = [0, 1, 0, 0, 0, 0]
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<1x2x1x1x4x1xf16> -> vector<16x16xf16>

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile        = [1, 1],
  thread_tile       = [8, 1],
  element_tile     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

func.func @distribute_transfer_read_row_major_with_nontrivial_index(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_row_major) : vector<16x16xf16>
  func.return %rootl : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL: @distribute_transfer_read_row_major_with_nontrivial_index
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[X:.+]]:2 = affine.delinearize_index %[[IDX]] into (8) : index, index
// CHECK: %[[OFF0:.+]] = affine.linearize_index [%[[X]]#1, %[[I0]]]  by (8, 1)
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF0]], %[[I1]]]
// CHECK: %[[OFF1:.+]] = affine.linearize_index [%c1, %[[I1]]] by (2, 8)
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF0]], %[[OFF1]]]
// CHECK: %[[OFF2:.+]] = affine.linearize_index [%c1, %[[X]]#1, %[[I0]]] by (2, 8, 1)
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF2]], %[[I1]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF2]], %[[OFF1]]]

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 2],
  outer_tile        = [1, 1],
  thread_tile       = [4, 8],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

func.func @distribute_transfer_read_col_major_with_broadcast(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (0, 0)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_col_major) : vector<16x16xf16>
  func.return %rootl : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (0, 0)>

// CHECK-LABEL: @distribute_transfer_read_col_major_with_broadcast
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[BROADCAST_READ:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[I1]]], %{{.*}} permutation_map = #[[$MAP]]
// CHECK: vector.insert_strided_slice %[[BROADCAST_READ]], %{{.*}} {offsets = [0, 0, 0, 0, 0, 0]
// CHECK: vector.insert_strided_slice %[[BROADCAST_READ]], %{{.*}} {offsets = [0, 1, 0, 0, 0, 0]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile        = [1, 1],
  thread_tile       = [8, 1],
  element_tile     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

// CHECK-DAG: #[[$PERM:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>

func.func @distribute_transfer_read_row_major_transpose(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_row_major) : vector<16x16xf16>
  func.return %rootl : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @distribute_transfer_read_row_major_transpose
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[X:.+]]:2 = affine.delinearize_index %[[IDX]] into (8) : index, index
// CHECK: %[[LIN_ID0:.+]] = affine.linearize_index [%[[X]]#1, %[[I1]]] by (8, 1)
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID0]]], {{.*}} permutation_map = #[[$PERM]]
// CHECK: %[[I0_PLUS_8:.+]] = affine.linearize_index [%c1, %[[I0]]] by (2, 8)
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0_PLUS_8]], %[[LIN_ID0]]], {{.*}} permutation_map = #[[$PERM]]
// CHECK: %[[LIN_ID1:.+]] = affine.linearize_index [%c1, %[[X]]#1, %[[I1]]] by (2, 8, 1)
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID1]]], {{.*}} permutation_map = #[[$PERM]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0_PLUS_8]], %[[LIN_ID1]]], %cst_0 {in_bounds = [true, true], permutation_map = #[[$PERM]]} : memref<32x32x32x32xf16>, vector<1x8xf16>

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 2],
  outer_tile        = [1, 1],
  thread_tile       = [4, 8],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>

// CHECK-LABEL: @distribute_transfer_read_col_major_transpose
func.func @distribute_transfer_read_col_major_transpose(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_col_major) : vector<16x16xf16>
  func.return %rootl : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.transfer_read {{.*}} permutation_map = #[[$MAP2]]
// CHECK: vector.transfer_read {{.*}} permutation_map = #[[$MAP2]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [7, 3, 1, 1],
  batch_tile    = [3, 5, 2, 1],
  outer_tile        = [1, 1, 2, 4],
  thread_tile       = [1, 1, 2, 2],
  element_tile     = [1, 1, 1, 2],

  subgroup_strides        = [3, 1, 1, 1],
  thread_strides          = [1, 1, 1, 2]
>

func.func @distribute_transfer_read_row_major_with_permutations(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<21x15x8x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true, true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d3, 0, d1)>}
                  : memref<32x32x32x32xf16>, vector<21x15x8x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<21x15x8x16xf16>
  func.return %rootl : vector<21x15x8x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL: @distribute_transfer_read_row_major_with_permutations

// Verify that there are (batch0: 3) * (batch1: 5) * (outer3: 4) = 60 total
// unique transfer read ops. The broadcasted dimension (2) CSEs the duplicate
// reads.
// CHECK-COUNT-60: vector.transfer_read

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [1],
  outer_tile        = [1],
  thread_tile       = [4],
  element_tile     = [4],

  subgroup_strides        = [1],
  thread_strides          = [16]
>

// CHECK-LABEL: @distribute_transfer_read_broadcast
func.func @distribute_transfer_read_broadcast(%arg0: memref<32x32xf16>) -> vector<16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [true]} : memref<32x32xf16>, vector<16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16xf16>
  func.return %rootl : vector<16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[YX:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 16)
// CHECK: %[[LANEY:.+]] = affine.linearize_index disjoint [%[[YX]]#1, %c0] by (4, 4)
// CHECK: %[[RD:.+]] = vector.transfer_read %{{.*}}[%c0, %[[LANEY:.+]]], {{.*}} : memref<32x32xf16>, vector<4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2],
  batch_tile    = [1],
  outer_tile        = [1],
  thread_tile       = [16],
  element_tile     = [4],

  subgroup_strides        = [1],
  thread_strides          = [1]
>

// CHECK-LABEL: @distribute_transfer_read_broadcast2
func.func @distribute_transfer_read_broadcast2(%arg0: memref<32x128xf16>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [true]} : memref<32x128xf16>, vector<128xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<128xf16>
  func.return %rootl : vector<128xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[YX:.+]]:3 = affine.delinearize_index %[[IDX]] into (2, 64)
// CHECK: %[[SUBGROUP:.+]]:2 = affine.delinearize_index %[[IDX]] into (16)
// CHECK: %[[LANEY:.+]] = affine.linearize_index disjoint [%[[YX]]#1, %[[SUBGROUP]]#1, %c0] by (2, 16, 4)
// CHECK: %[[RD:.+]] = vector.transfer_read %{{.*}}[%c0, %[[LANEY:.+]]], {{.*}} : memref<32x128xf16>, vector<4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [],
  batch_tile    = [],
  outer_tile        = [],
  thread_tile       = [],
  element_tile     = [],

  subgroup_strides        = [],
  thread_strides          = []
>

// CHECK-LABEL: @distribute_transfer_read_0d
func.func @distribute_transfer_read_0d(%arg0: memref<128xf16>) -> vector<f16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0], %cst
          {in_bounds = []} : memref<128xf16>, vector<f16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<f16>
  func.return %rootl : vector<f16>
}


builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[RD:.+]] = vector.transfer_read %{{.*}}[%c0]
// CHECK-SAME: memref<128xf16>, vector<f16>
// CHECK: iree_vector_ext.to_simd %[[RD]]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile        = [1, 1],
  thread_tile       = [8, 1],
  element_tile     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

// CHECK-LABEL: @distribute_transfer_write_row_major
func.func @distribute_transfer_write_row_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_row_major) : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0]
          {in_bounds = [true, true]}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LANEX:.+]]:2 = affine.delinearize_index %[[IDX]] into (8)
// CHECK: %[[SLICE:.+]] = vector.extract %{{.*}}[0, 0, 0, 0] : vector<1x8xf16> from vector<2x2x1x1x1x8xf16>
// CHECK: vector.transfer_write %[[SLICE]], %{{.*}}[%[[LANEX]]#1, %c0] {in_bounds = [true, true]} : vector<1x8xf16>, memref<64x64xf16>
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[LANEX]]#1, %c8]
// CHECK: %[[LANEX_PLUS_VECDIMX:.+]] = affine.linearize_index disjoint [%c1, %[[LANEX]]#1] by (2, 8)
// CHECK: vector.extract %{{.*}}[1, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEX_PLUS_VECDIMX]], %c0]
// CHECK: vector.extract %{{.*}}[1, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEX_PLUS_VECDIMX]], %c8]

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 2],
  outer_tile        = [1, 1],
  thread_tile       = [4, 8],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [8, 1]
>

// CHECK-LABEL: @distribute_transfer_write_col_major
func.func @distribute_transfer_write_col_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_col_major) : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0]
          {in_bounds = [true, true]}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[YX:.+]]:3 = affine.delinearize_index %[[IDX]] into (4, 8)
// CHECK: %[[LANEY:.+]] = affine.linearize_index disjoint [%[[YX]]#1, %c0] by (4, 4)
// CHECK: vector.extract %{{.*}}[0, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEY]], %[[YX]]#2]
// CHECK: %[[LANEX:.+]] = affine.linearize_index disjoint [%c1, %[[YX]]#2] by (2, 8)
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write {{.*}}[%[[LANEY]], %[[LANEX]]]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile        = [1, 1],
  thread_tile       = [8, 1],
  element_tile     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>

func.func @distribute_transfer_write_row_major_with_nontrivial_index(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_row_major) : vector<16x16xf16>
  vector.transfer_write %rootl, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @distribute_transfer_write_row_major_with_nontrivial_index
// CHECK-SAME:    vector<16x16xf16>, %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[LANE:.+]]:2 = affine.delinearize_index %[[IDX]] into (8)
// CHECK: %[[LIN_ID0:.+]] = affine.linearize_index [%[[LANE]]#1, %[[I1]]] by (8, 1)
// CHECK: vector.extract %{{.*}}[0, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID0]]] {{.*}} permutation_map = #[[$MAP]]
// CHECK: %[[LIN_ID1:.+]] = affine.linearize_index [%c1, %[[I0]]] by (2, 8)
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[LIN_ID1]], %[[LIN_ID0]]] {{.*}} permutation_map = #[[$MAP]]
// CHECK: %[[LIN_ID2:.+]] = affine.linearize_index [%c1, %[[LANE]]#1, %[[I1]]]
// CHECK: vector.extract %{{.*}}[1, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID2]]] {{.*}} permutation_map = #[[$MAP]]
// CHECK: vector.extract %{{.*}}[1, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[LIN_ID1]], %[[LIN_ID2]]] {{.*}} permutation_map = #[[$MAP]]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile        = [1, 1],
  thread_tile       = [8, 1],
  element_tile     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

func.func @distribute_transfer_read_write(%a: index, %b: index,
                                          %arg0: memref<32x32x32x32xf16>,
                                          %arg1: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>

  %rootl = iree_vector_ext.to_layout %root to layout(#layout_row_major) : vector<16x16xf16>

  vector.transfer_write %rootl, %arg1[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[B00:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX:[a-zA-Z0-9]+]], %[[OFFSET0:[a-zA-Z0-9]+]]]
// CHECK: %[[B10:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET1:[a-zA-Z0-9]+]]]
// CHECK: %[[B01:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH:[a-zA-Z0-9]+]], %[[OFFSET0]]]
// CHECK: %[[B11:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET1]]]
// CHECK: vector.transfer_write %[[B00]], %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET0]]]
// CHECK: vector.transfer_write %[[B10]], %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET1]]]
// CHECK: vector.transfer_write %[[B01]], %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET0]]]
// CHECK: vector.transfer_write %[[B11]], %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET1]]]

// -----

// A: shape = 128x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [4, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [2, 1],
  thread_strides          = [1, 32]
>

// B: shape = 8x64, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 2],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

// C: shape = 128x64, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [4, 2],
  batch_tile    = [1, 1],
  outer_tile        = [4, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [2, 1],
  thread_strides          = [32, 1]
>

// CHECK-LABEL: @mfma_64x128x8_read
func.func @mfma_64x128x8_read(%mem: memref<128x8xf16>,
                              %mem1: memref<8x64xf16>,
                              %mem2: memref<128x64xf16>)
                -> (vector<128x8xf16>, vector<8x64xf16>, vector<128x64xf16>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16

  // CHECK: %[[IDX:.+]] = gpu.thread_id  x
  // CHECK-DAG: %[[WG:.+]]:4 = affine.delinearize_index %[[IDX]] into (4, 2, 64)
  // CHECK-DAG: %[[LANE:.+]]:3 = affine.delinearize_index %[[IDX]] into (2, 32)
  // This doesn't canonicalize away currently, but could be equivalent to %WG
  // CHECK-DAG: %[[WG_N:.+]]:3 = affine.delinearize_index %[[IDX]] into (2, 64)
  // CHECK-DAG: %[[LHSM:.+]] = affine.linearize_index disjoint [%[[WG]]#1, %[[LANE]]#2]
  // LHSK = RHSK
  // CHECK-DAG: %[[LHSK:.+]] = affine.linearize_index disjoint [%[[LANE]]#1, %c0] by (2, 4)
  // ACCN = RHSN
  // CHECK-DAG: %[[RHSN_DUP_WG:.+]] = affine.linearize_index disjoint [%[[WG_N]]#1, %[[LANE]]#2] by (2, 32)
  // CHECK-DAG: %[[RHSN:.+]] = affine.linearize_index disjoint [%[[WG]]#2, %[[LANE]]#2] by (2, 32)

  // M is unrolled 4 times.
  // CHECK-DAG: %[[ACCM0:.+]] = affine.linearize_index disjoint [%[[WG]]#1, %c0, %[[LANE]]#1, %c0] by (4, 4, 2, 4)
  // CHECK-DAG: %[[ACCM1:.+]] = affine.linearize_index disjoint [%[[WG]]#1, %c1, %[[LANE]]#1, %c0] by (4, 4, 2, 4)
  // CHECK-DAG: %[[ACCM2:.+]] = affine.linearize_index disjoint [%[[WG]]#1, %c2, %[[LANE]]#1, %c0] by (4, 4, 2, 4)
  // CHECK-DAG: %[[ACCM3:.+]] = affine.linearize_index disjoint [%[[WG]]#1, %c3, %[[LANE]]#1, %c0] by (4, 4, 2, 4)

  // M, K
  // CHECK-DAG: transfer_read %{{.*}}[%[[LHSM]], %[[LHSK]]]
  // K, N
  // CHECK-DAG: transfer_read %{{.*}}[%[[LHSK]], %[[RHSN_DUP_WG]]]
  // M, N
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM0]], %[[RHSN]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM1]], %[[RHSN]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM2]], %[[RHSN]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[ACCM3]], %[[RHSN]]

  %a = vector.transfer_read %mem[%c0, %c0], %cst
          {in_bounds = [true, true]}
  : memref<128x8xf16>, vector<128x8xf16>
  %b = vector.transfer_read %mem1[%c0, %c0], %cst
          {in_bounds = [true, true]}
  : memref<8x64xf16>, vector<8x64xf16>
  %c = vector.transfer_read %mem2[%c0, %c0], %cst
          {in_bounds = [true, true]}
  : memref<128x64xf16>, vector<128x64xf16>

  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<128x8xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<8x64xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<128x64xf16>

  return %A, %B, %C : vector<128x8xf16>, vector<8x64xf16>, vector<128x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [2, 1],
  thread_strides          = [1, 32]
>

func.func @transposed_read_64x8(%mem: memref<8x64xf16>)
                -> (vector<64x8xf16>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16

  %read = vector.transfer_read %mem[%c0, %c0], %cst
          {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
  : memref<8x64xf16>, vector<64x8xf16>
  %readl = iree_vector_ext.to_layout %read to layout(#layout) : vector<64x8xf16>

  return %readl : vector<64x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @transposed_read_64x8

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK-DAG: %[[WG:.+]]:4 = affine.delinearize_index %[[IDX]] into (2, 2, 64)
// CHECK-DAG: %[[LANE:.+]]:3 = affine.delinearize_index %[[IDX]] into (2, 32)
// CHECK-DAG: %[[M:.+]] = affine.linearize_index disjoint [%[[WG]]#1, %[[LANE]]#2] by (2, 32)
// CHECK-DAG: %[[N:.+]] = affine.linearize_index disjoint [%[[LANE]]#1, %c0] by (2, 4)
// CHECK: transfer_read %{{.*}}[%[[N]], %[[M]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2],
  batch_tile = [2, 4],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],
  subgroup_strides = [2, 1],
  thread_strides   = [16, 1]
>

func.func @broadcast(%src: vector<128xf16>) -> (vector<64x128xf16>) {
  %bcast = vector.broadcast %src
    : vector<128xf16> to vector<64x128xf16>
  %bcastl = iree_vector_ext.to_layout %bcast to layout(#layout) : vector<64x128xf16>
  return %bcastl : vector<64x128xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[BROAD:.+]] = vector.broadcast %{{.*}} : vector<4x1x1xf16> to vector<2x1x4x4x1x1xf16>
// CHECK: %[[TRANS:.+]] = vector.transpose %[[BROAD]], [0, 3, 1, 4, 2, 5]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2, 2],
  batch_tile = [2, 2, 1],
  outer_tile = [2, 1, 1],
  thread_tile = [4, 16, 8],
  element_tile = [1, 4, 4],
  subgroup_strides = [4, 2, 1],
  thread_strides   = [128, 8, 1]
>

func.func @broadcast(%src: vector<64xf16>) -> (vector<32x256x64xf16>) {
  %bcast = vector.broadcast %src
    : vector<64xf16> to vector<32x256x64xf16>
  %bcastl = iree_vector_ext.to_layout %bcast to layout(#layout) : vector<32x256x64xf16>
  return %bcastl : vector<32x256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.broadcast %{{.*}} : vector<1x1x4xf16> to vector<2x2x1x2x1x1x1x4x4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2, 2],
  batch_tile = [2, 2, 1],
  outer_tile = [2, 1, 1],
  thread_tile = [4, 16, 8],
  element_tile = [1, 4, 4],
  subgroup_strides = [4, 2, 1],
  thread_strides = [128, 8, 1]
>

func.func @scalar_broadcast(%src: f16) -> (vector<32x256x64xf16>) {
  %bcast = vector.broadcast %src : f16 to vector<32x256x64xf16>
  %bcastl = iree_vector_ext.to_layout %bcast to layout(#layout) : vector<32x256x64xf16>
  return %bcastl : vector<32x256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.broadcast %{{.*}} : f16 to vector<2x2x1x2x1x1x1x4x4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2, 2],
  batch_tile = [2, 2, 1],
  outer_tile = [2, 1, 1],
  thread_tile = [4, 16, 8],
  element_tile = [1, 4, 4],
  subgroup_strides = [4, 2, 1],
  thread_strides = [128, 8, 1]
>

func.func @zero_rank_broadcast(%src: vector<f16>) -> (vector<32x256x64xf16>) {
  %bcast = vector.broadcast %src : vector<f16> to vector<32x256x64xf16>
  %bcastl = iree_vector_ext.to_layout %bcast to layout(#layout) : vector<32x256x64xf16>
  return %bcastl : vector<32x256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.broadcast %{{.*}} : vector<f16> to vector<2x2x1x2x1x1x1x4x4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2, 2],
  batch_tile = [2, 2, 1],
  outer_tile = [2, 1, 1],
  thread_tile = [4, 16, 8],
  element_tile = [1, 4, 4],
  subgroup_strides = [4, 2, 1],
  thread_strides   = [128, 8, 1]
>

func.func @gather(%base: memref<32x256x64xf16>, %index_vec: vector<32x256x64xindex>)-> (vector<32x256x64xf16>){
  %c0 = arith.constant 0 : index
  %mask = arith.constant dense<true> : vector<32x256x64xi1>
  %pass = arith.constant dense<0.000000e+00> : vector<32x256x64xf16>
  %0 = vector.gather %base[%c0, %c0, %c0] [%index_vec], %mask, %pass : memref<32x256x64xf16>, vector<32x256x64xindex>, vector<32x256x64xi1>, vector<32x256x64xf16> into vector<32x256x64xf16>
  %1 = iree_vector_ext.to_layout %0 to layout(#layout) :  vector<32x256x64xf16>
  return %1 : vector<32x256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @gather
// CHECK-SAME:  (%[[SRC:.*]]: memref<32x256x64xf16>, %[[INDEX:.*]]: vector<32x256x64xindex>)
// CHECK: %[[DIST_INDEX:.*]] = iree_vector_ext.to_simt %[[INDEX]] : vector<32x256x64xindex> -> vector<2x2x1x2x1x1x1x4x4xindex>
// CHECK: %[[GATHER:.*]] = vector.gather %[[SRC]][%c0, %c0, %c0] [%[[DIST_INDEX]]]
// CHECK: iree_vector_ext.to_simd %[[GATHER]] : vector<2x2x1x2x1x1x1x4x4xf16> -> vector<32x256x64xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2],
  batch_tile = [2, 4],
  outer_tile = [2, 1],
  thread_tile = [4, 16],
  element_tile = [2, 2],
  subgroup_strides = [2, 1],
  thread_strides = [16, 1]
>

func.func @transpose(%src: vector<256x64xf16>) -> (vector<64x256xf16>) {
  %transp = vector.transpose %src, [1, 0]
    : vector<256x64xf16> to vector<64x256xf16>
  %transpl = iree_vector_ext.to_layout %transp to layout(#layout) : vector<64x256xf16>
  %sqrt = math.sqrt %transpl : vector<64x256xf16>
  return %sqrt : vector<64x256xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @transpose
// CHECK: iree_vector_ext.to_simt %{{.*}} : vector<256x64xf16> -> vector<4x2x1x2x2x2xf16>
// CHECK: vector.transpose %{{.*}}, [1, 0, 3, 2, 5, 4] : vector<4x2x1x2x2x2xf16> to vector<2x4x2x1x2x2xf16>
// CHECK: math.sqrt %{{.*}} : vector<2x4x2x1x2x2xf16>
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x4x2x1x2x2xf16> -> vector<64x256xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2],
  batch_tile = [2, 4],
  outer_tile = [2, 1],
  thread_tile = [4, 16],
  element_tile = [2, 2],
  subgroup_strides = [2, 1],
  thread_strides = [16, 1]
>

func.func @transpose(%src: vector<64x256xf16>) -> (vector<256x64xf16>) {
  %srcl = iree_vector_ext.to_layout %src to layout(#layout) : vector<64x256xf16>
  %transp = vector.transpose %srcl, [1, 0]
    : vector<64x256xf16> to vector<256x64xf16>
  %sqrt = math.sqrt %transp : vector<256x64xf16>
  return %sqrt : vector<256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @transpose
// CHECK: iree_vector_ext.to_simt %{{.*}} : vector<64x256xf16> -> vector<2x4x2x1x2x2xf16>
// CHECK: vector.transpose %{{.*}}, [1, 0, 3, 2, 5, 4] : vector<2x4x2x1x2x2xf16> to vector<4x2x1x2x2x2xf16>
// CHECK: math.sqrt %{{.*}} : vector<4x2x1x2x2x2xf16>
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<4x2x1x2x2x2xf16> -> vector<256x64xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1, 1],
  batch_tile = [1, 2, 4],
  outer_tile = [1, 1, 1],
  thread_tile = [4, 8, 2],
  element_tile = [4, 1, 2],

  subgroup_strides = [1, 1, 1],
  thread_strides = [16, 2, 1]
>

func.func @transpose_3d(%arr: memref<32x32x32xf16>) -> () {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.0 : f16
  %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
  %root = vector.transfer_read %arr[%c0, %c0, %c0], %cst_0 {
    in_bounds = [true, true, true]
  } : memref<32x32x32xf16>, vector<32x16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<32x16x16xf16>
  %t = vector.transpose %rootl, [1, 2, 0] : vector<32x16x16xf16> to vector<16x16x32xf16>
  vector.transfer_write %t, %arr[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<16x16x32xf16>, memref<32x32x32xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @transpose_3d
// CHECK-DAG:         %[[IDX:.+]] = gpu.thread_id  x
// CHECK-DAG:         %[[WG:.+]]:3 = affine.delinearize_index %[[IDX]] into (2, 64)
// CHECK-DAG:         %[[LANE:.+]]:4 = affine.delinearize_index %[[IDX]] into (4, 8, 2)
// CHECK-DAG:         %[[DIM:.+]]  = affine.linearize_index disjoint [%[[WG]]#1, %[[LANE]]#1, %c0] by (2, 4, 4)
//       COM:           DIM1 == LANE#2
// CHECK-DAG:         %[[DIM2:.+]] = affine.linearize_index disjoint [%[[LANE]]#3, %c0] by (2, 2)
// CHECK-DAG:         %[[DIM3:.+]] = affine.linearize_index disjoint [%c1, %[[LANE]]#3, %c0] by (4, 2, 2)
// CHECK-DAG:         %[[DIM4:.+]] = affine.linearize_index disjoint [%c2, %[[LANE]]#3, %c0] by (4, 2, 2)
// CHECK-DAG:         %[[DIM5:.+]] = affine.linearize_index disjoint [%c3, %[[LANE]]#3, %c0] by (4, 2, 2)
// CHECK-DAG:         %[[DIM6:.+]] = affine.linearize_index disjoint [%c1, %[[LANE]]#2] by (2, 8)
// CHECK-DAG:         %[[RD0:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[LANE]]#2, %[[DIM2]]], {{.*}} : memref<32x32x32xf16>, vector<4x1x2xf16>
// CHECK-DAG:         %[[RD1:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[LANE]]#2, %[[DIM3]]]
// CHECK-DAG:         %[[RD2:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[LANE]]#2, %[[DIM4]]]
// CHECK-DAG:         %[[RD3:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[LANE]]#2, %[[DIM5]]]
// CHECK-DAG:         %[[RD4:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM2]]]
// CHECK-DAG:         %[[RD5:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM3]]]
// CHECK-DAG:         %[[RD6:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM4]]]
// CHECK-DAG:         %[[RD7:.+]] = vector.transfer_read %arg0[%[[DIM]], %[[DIM6]], %[[DIM5]]]

// CHECK:         vector.transpose %{{.*}}, [1, 2, 0, 4, 5, 3, 7, 8, 6] : vector<1x2x4x1x1x1x4x1x2xf16> to vector<2x4x1x1x1x1x1x2x4xf16>

// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[LANE]]#2, %[[DIM2]], %[[DIM]]] {{.*}} : vector<1x2x4xf16>, memref<32x32x32xf16>
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[LANE]]#2, %[[DIM3]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[LANE]]#2, %[[DIM4]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[LANE]]#2, %[[DIM5]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM2]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM3]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM4]], %[[DIM]]]
// CHECK-DAG:         vector.transfer_write %{{.*}}, %arg0[%[[DIM6]], %[[DIM5]], %[[DIM]]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @distribute_scf_for(%arr: memref<32x32xf16>, %a: vector<32x32xf16>) -> vector<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<0.000000e+00> : vector<f32>
  %cst_0 = arith.constant 0.0 : f16
  %out = scf.for %i = %c0 to %c128 step %c1 iter_args(%arg0 = %cst) -> (vector<f32>) {
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<32x32xf16>
    %b = arith.addf %rootl, %a : vector<32x32xf16>
    %c = arith.extf %b : vector<32x32xf16> to vector<32x32xf32>
    %init = vector.extract %arg0[] : f32 from vector<f32>
    %root_red = vector.multi_reduction<add>, %c, %init [0, 1]  : vector<32x32xf32> to f32
    %d = vector.broadcast %root_red : f32 to vector<f32>
    scf.yield %d : vector<f32>
  }
  return %out : vector<f32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @distribute_scf_for
// CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
// CHECK: iter_args(%[[ARG0:.*]] = %[[ROOT]]) -> (vector<f32>)
// CHECK: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32x32xf16> -> vector<2x2x1x1x1x4xf16>
// CHECK: %[[B:.*]] = arith.addf %{{.*}}, %[[A]]
// CHECK: %[[C:.*]] = arith.extf %[[B]]
// CHECK-NEXT: %[[D:.*]] = vector.extract %[[ARG0]][]
// Local reduction
// CHECK: vector.multi_reduction <add>, %[[C]], %{{.*}} [0, 1, 2, 3, 4, 5] : vector<2x2x1x1x1x4xf32> to f32
// Global reduction
// CHECK: gpu.subgroup_reduce add %{{.*}} cluster(size = 16) : (f32) -> f32
// CHECK-NEXT: gpu.subgroup_reduce add %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// Accumulator reduction
// CHECK: vector.broadcast %[[D]] : f32 to vector<1xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<1xf32>

// -----

#contraction_accesses = [
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> ()>
]

#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["reduction", "reduction"],
  kind = #vector.kind<maxnumf>
}

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @contraction_32x32_alldims(%arg0: vector<32x32xf32>, %arg1: f32) -> f32 {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = vector.contract #contraction_trait %arg0l, %arg0l, %arg1 : vector<32x32xf32>, vector<32x32xf32> into f32
  return %0 : f32
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @contraction_32x32_alldims
// Local contraction
// CHECK: vector.contract {{.*}} vector<2x2x1x1x1x4xf32>, vector<2x2x1x1x1x4xf32> into f32
// Global reduction
// CHECK: gpu.subgroup_reduce maxnumf %{{.*}} cluster(size = 16) : (f32) -> f32
// CHECK-NEXT: gpu.subgroup_reduce maxnumf %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// Accumulator reduction
// CHECK: arith.maxnumf %{{.*}}, %{{.*}} : vector<1xf32>

// -----

#contraction_accesses = [
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> ()>
]

#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["reduction", "reduction"],
  kind = #vector.kind<maxnumf>
}

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @distribute_scf_for_contraction(%arr: memref<32x32xf16>, %a: vector<32x32xf16>) -> vector<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<0.000000e+00> : vector<f32>
  %cst_0 = arith.constant 0.0 : f16
  %out = scf.for %i = %c0 to %c128 step %c1 iter_args(%arg0 = %cst) -> (vector<f32>) {
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<32x32xf16>
    %b = arith.addf %rootl, %a : vector<32x32xf16>
    %c = arith.extf %b : vector<32x32xf16> to vector<32x32xf32>
    %init = vector.extract %arg0[] : f32 from vector<f32>
    %root_red = vector.contract #contraction_trait %c, %c,  %init : vector<32x32xf32>, vector<32x32xf32> into f32
    %d = vector.broadcast %root_red : f32 to vector<f32>
    scf.yield %d : vector<f32>
  }
  return %out : vector<f32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @distribute_scf_for_contraction
// CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
// CHECK: iter_args(%[[ARG0:.*]] = %[[ROOT]]) -> (vector<f32>)
// CHECK: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32x32xf16> -> vector<2x2x1x1x1x4xf16>
// CHECK: %[[B:.*]] = arith.addf %{{.*}}, %[[A]]
// CHECK: %[[C:.*]] = arith.extf %[[B]]
// CHECK-NEXT: %[[D:.*]] = vector.extract %[[ARG0]][]
// Local contraction
// CHECK: vector.contract {{.*}} vector<2x2x1x1x1x4xf32>, vector<2x2x1x1x1x4xf32> into f32
// Global reduction
// CHECK: gpu.subgroup_reduce maxnumf %{{.*}} cluster(size = 16) : (f32) -> f32
// CHECK-NEXT: gpu.subgroup_reduce maxnumf %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// Accumulator reduction
// CHECK: arith.maxnumf %{{.*}}, %{{.*}} : vector<1xf32>

// -----

#contraction_accesses = [
  affine_map<(m, k) -> (m, k)>,
  affine_map<(m, k) -> (k)>,
  affine_map<(m, k) -> (m)>
]

#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "reduction"],
  kind = #vector.kind<maxnumf>
}

#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [2],
  outer_tile = [1],
  thread_tile = [4],
  element_tile = [4],

  subgroup_strides = [1],
  thread_strides = [16]
>

#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [2],
  outer_tile = [1],
  thread_tile = [16],
  element_tile = [1],

  subgroup_strides = [1],
  thread_strides = [1]
>

func.func @contraction_dim1(%a: vector<32x32xf32>, %b: vector<32xf32>,  %init: vector<32xf32>) -> vector<32xf32> {
  %al = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<32x32xf32>
  %bl = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<32xf32>
  %output = vector.contract #contraction_trait %al, %bl, %init : vector<32x32xf32>, vector<32xf32>, vector<32xf32> into vector<32xf32>
  %0 = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @contraction_dim1
// Local contraction
// CHECK: vector.contract {{.*}} vector<2x2x1x1x1x4xf32>, vector<2x1x4xf32> into vector<2x1x1xf32>
// Global reduction
// CHECK: vector.extract %{{.*}}[0, 0, 0]
// CHECK-NEXT: gpu.subgroup_reduce maxnumf %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// CHECK: vector.extract %{{.*}}[1, 0, 0]
// CHECK-NEXT: gpu.subgroup_reduce maxnumf %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// Accumulator reduction
// CHECK: arith.maxnumf %{{.*}}, %{{.*}} : vector<2x1x1xf32>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [32],
  element_tile = [2],

  subgroup_strides = [1],
  thread_strides = [1]
>

func.func @zero_d_vector_extract(%vec : vector<64xf32>, %acc : vector<f32>) -> f32 {
  %layouted = iree_vector_ext.to_layout %vec to layout(#layout) : vector<64xf32>
  %scalar = vector.extract %acc[] : f32 from vector<f32>
  %out = vector.multi_reduction <add>, %layouted, %scalar [0] : vector<64xf32> to f32
  return %out : f32
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @zero_d_vector_extract
// CHECK-SAME:      %[[VEC:.+]]: vector<64xf32>, %[[ACC:.+]]: vector<f32>
// CHECK-DAG:  %[[SIMT_ACC:.+]] = iree_vector_ext.to_simt %[[ACC]] : vector<f32> -> vector<f32>
// CHECK-DAG:  %[[SCALAR:.+]] = vector.extract %[[SIMT_ACC]][] : f32 from vector<f32>
// CHECK-DAG:  %[[SIMT:.+]] = iree_vector_ext.to_simt %[[VEC]] : vector<64xf32> -> vector<1x1x2xf32>
// CHECK:      %[[LOCAL:.+]] = vector.multi_reduction <add>, %[[SIMT]], %{{.*}}
// CHECK:      gpu.subgroup_reduce add %[[LOCAL]]
// Accumulator addition
// CHECK:      %[[BROADCASTED:.+]] = vector.broadcast %[[SCALAR]] : f32 to vector<1xf32>
// CHECK:      arith.addf %{{.*}}, %[[BROADCASTED]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [4, 1],
  batch_tile = [4, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [1, 8],

  subgroup_strides = [1, 0],
  thread_strides = [0, 0]
>

func.func @paged_transfer_gather(%indices: vector<16xindex>,
  %source: memref<4096x512x8xf16>) -> vector<16x8xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %dim = memref.dim %source, %c0 : memref<4096x512x8xf16>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0, %c0]
  [None, %indices: vector<16xindex>, None], %cst0 { indexed_maps = [
                                             affine_map<(d0, d1, d2) -> (d1)>],
    permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>,
    in_bounds = [true, true] }
  : memref<4096x512x8xf16>, vector<16x8xf16>

  %l_out = iree_vector_ext.to_layout %out to layout(#layout) : vector<16x8xf16>

  return %l_out : vector<16x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @paged_transfer_gather
// CHECK-SAME: %[[INDICES:.+]]: vector<16xindex>, %[[SOURCE:.+]]: memref<4096x512x8xf16>
// CHECK: %[[DIS_INDICES:.+]] = iree_vector_ext.to_simt %[[INDICES]] : vector<16xindex> -> vector<4x1x1xindex>
// CHECK: %[[GATHER0:.+]] = vector.extract %[[DIS_INDICES]][0, 0, 0]
// CHECK: vector.transfer_read %[[SOURCE]][%c0, %[[GATHER0]], %c0]
// CHECK: %[[GATHER1:.+]] = vector.extract %[[DIS_INDICES]][1, 0, 0]
// CHECK: vector.transfer_read %[[SOURCE]][%c0, %[[GATHER1]], %c0]
// CHECK: %[[GATHER2:.+]] = vector.extract %[[DIS_INDICES]][2, 0, 0]
// CHECK: vector.transfer_read %[[SOURCE]][%c0, %[[GATHER2]], %c0]
// CHECK: %[[GATHER3:.+]] = vector.extract %[[DIS_INDICES]][3, 0, 0]
// CHECK: vector.transfer_read %[[SOURCE]][%c0, %[[GATHER3]], %c0]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [4, 1],
  batch_tile = [4, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [1, 8],

  subgroup_strides = [1, 0],
  thread_strides = [0, 0]
>

func.func @paged_transfer_gather_multi_index(%indices: vector<16xindex>,
  %indices2: vector<8x16xindex>,
  %source: memref<4096x512x8xf16>) -> vector<16x8xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %dim = memref.dim %source, %c0 : memref<4096x512x8xf16>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0, %c0]
  [None, %indices: vector<16xindex>, %indices2: vector<8x16xindex>], %cst0
                                           { indexed_maps = [
                                             affine_map<(d0, d1, d2) -> (d1)>,
                                             affine_map<(d0, d1, d2) -> (d2, d1)>],
    permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>,
    in_bounds = [true, true] }
  : memref<4096x512x8xf16>, vector<16x8xf16>

  %l_out = iree_vector_ext.to_layout %out to layout(#layout) : vector<16x8xf16>

  return %l_out : vector<16x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @paged_transfer_gather_multi_index
// CHECK-COUNT-4: vector_ext.transfer_gather

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile        = [1, 1],
  thread_tile       = [8, 1],
  element_tile     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 1]
>

func.func @distribute_map_scatter_row_major(%root: vector<16x16xf16>, %output: memref<64x64xf16>) {
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_row_major) : vector<16x16xf16>
  iree_linalg_ext.map_scatter %rootl into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : vector<16x16xf16> into memref<64x64xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @distribute_map_scatter_row_major
//   CHECK-DAG:   %[[IDX:.+]] = gpu.thread_id  x
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[LANEX:.+]]:2 = affine.delinearize_index %[[IDX]] into (8)
//   CHECK-DAG:   %[[SLICE0:.+]] = vector.extract %{{.*}}[0, 0, 0, 0]
//       CHECK:   iree_linalg_ext.map_scatter %[[SLICE0]]
//       CHECK:     ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:       %[[DISTRIBUTED_IDX0:.+]] = arith.addi %[[IDX0]], %[[LANEX]]#1
//       CHECK:       iree_linalg_ext.yield %[[DISTRIBUTED_IDX0]], %[[IDX1]]
//       CHECK:     : vector<1x8xf16> into memref<64x64xf16>
//       CHECK:   %[[SLICE1:.+]] = vector.extract %{{.*}}[0, 1, 0, 0]
//       CHECK:   iree_linalg_ext.map_scatter %[[SLICE1]]
//       CHECK:     ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//   CHECK-DAG:       %[[DISTRIBUTED_IDX0:.+]] = arith.addi %[[IDX0]], %[[LANEX]]#1
//   CHECK-DAG:       %[[DISTRIBUTED_IDX1:.+]] = arith.addi %[[IDX1]], %[[C8]]
//       CHECK:       iree_linalg_ext.yield %[[DISTRIBUTED_IDX0]], %[[DISTRIBUTED_IDX1]]
//       CHECK:     : vector<1x8xf16> into memref<64x64xf16>
//   CHECK-DAG:   %[[LANEX_PLUS_VECDIMX:.+]] = affine.linearize_index disjoint [%c1, %[[LANEX]]#1] by (2, 8)
//   CHECK-DAG:   %[[SLICE2:.+]] = vector.extract %{{.*}}[1, 0, 0, 0]
//       CHECK:   iree_linalg_ext.map_scatter %[[SLICE2]]
//       CHECK:     ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:       %[[DISTRIBUTED_IDX0:.+]] = arith.addi %[[IDX0]], %[[LANEX_PLUS_VECDIMX]]
//       CHECK:       iree_linalg_ext.yield %[[DISTRIBUTED_IDX0]], %[[IDX1]]
//       CHECK:     : vector<1x8xf16> into memref<64x64xf16>
//       CHECK:   %[[SLICE3:.+]] = vector.extract %{{.*}}[1, 1, 0, 0]
//       CHECK:   iree_linalg_ext.map_scatter %[[SLICE3]]
//       CHECK:     ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//   CHECK-DAG:       %[[DISTRIBUTED_IDX0:.+]] = arith.addi %[[IDX0]], %[[LANEX_PLUS_VECDIMX]]
//   CHECK-DAG:       %[[DISTRIBUTED_IDX1:.+]] = arith.addi %[[IDX1]], %[[C8]]
//       CHECK:       iree_linalg_ext.yield %[[DISTRIBUTED_IDX0]], %[[DISTRIBUTED_IDX1]]
//       CHECK:     : vector<1x8xf16> into memref<64x64xf16>
