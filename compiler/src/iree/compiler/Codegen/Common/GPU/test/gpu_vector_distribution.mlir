// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

#layout = #iree_vector_ext.layout<<[VECTORY, LANEY], [4, 4]>, <[VECTORX, LANEX], [4, 4]>>

// CHECK-LABEL: @distribute_elementwise_f16
func.func @distribute_elementwise_f16(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.0 : f16
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
  %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout} dense<0.0> : vector<16x16xf16>
  // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<16xf16>
  // CHECK-DAG: %[[C:.*]] = arith.mulf %[[B]], %[[ROOT]] : vector<16xf16>
  %c = arith.mulf %root, %b : vector<16x16xf16>
  // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<16xf16>
  // CHECK-DAG: %[[D:.*]] = arith.addf %[[C]], %[[A]] fastmath<reassoc,nnan> : vector<16xf16>
  %d = arith.addf %c, %a fastmath<reassoc,nnan> : vector<16x16xf16>
  // CHECK: iree_vector_ext.to_simd %[[D]] : vector<16xf16> -> vector<16x16xf16>
  return %d : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_elementwise_i32
func.func @distribute_elementwise_i32(%a: vector<16x16xi32>, %b: vector<16x16xi32>) -> vector<16x16xi32> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0 : i32
  // CHECK: %[[ROOT:.*]] = arith.constant dense<2> : vector<16xi32>
  %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout} dense<2> : vector<16x16xi32>
  // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
  // CHECK-DAG: %[[C:.*]] = arith.muli %[[B]], %[[ROOT]] : vector<16xi32>
  %c = arith.muli %root, %b : vector<16x16xi32>
  // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
  // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] : vector<16xi32>
  %d = arith.addi %c, %a : vector<16x16xi32>
  // CHECK: iree_vector_ext.to_simd %[[D]] : vector<16xi32> -> vector<16x16xi32>
  return %d : vector<16x16xi32>
}

func.func @distribute_scf_for(%a: vector<16x16xi32>, %b: vector<16x16xi32>) -> vector<16x16xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_0 = arith.constant 0 : i32
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0> : vector<16xi32>
  %root = arith.constant {"__vector_layout_test_anchor_result_0" = #layout} dense<0> : vector<16x16xi32>
  // CHECK: iter_args(%[[ARG0:.*]] = %[[ROOT]]) -> (vector<16xi32>)
  %out = scf.for %i = %c0 to %c128 step %c1 iter_args(%arg0 = %root) -> (vector<16x16xi32>) {
    // These should be ideally folded if canonicalization was ever ran.
    // Canonicalization currently breaks other tests. If canonicalization
    // is ever ran, this should be updated.
    // CHECK-DAG: %[[TMP:.*]] = iree_vector_ext.to_simd %[[ARG0]] : vector<16xi32> -> vector<16x16xi32> 
    // CHECK-DAG: %[[ARG0S:.*]] = iree_vector_ext.to_simt %[[TMP]] : vector<16x16xi32> -> vector<16xi32>
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
    // CHECK-DAG: %[[C:.*]] = arith.muli %[[ARG0S]], %[[B]] : vector<16xi32>
    %c = arith.muli %arg0, %b : vector<16x16xi32>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<16xi32>
    // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] : vector<16xi32>
    %d = arith.addi %c, %a : vector<16x16xi32>
    // CHECK: scf.yield %[[D]] : vector<16xi32>
    scf.yield %d : vector<16x16xi32>
  }
  return %out : vector<16x16xi32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_row_major = #iree_vector_ext.layout<<[BATCHX, LANEY], [2, 8]>, <[BATCHY, LANEX, VECTORX], [2, 1, 8]>>
#layout_col_major = #iree_vector_ext.layout<<[BATCHX, LANEY, VECTORX], [1, 4, 4]>, <[BATCHY, LANEX], [2, 8]>>

// CHECK-LABEL: @distribute_transfer_read_row_major
func.func @distribute_transfer_read_row_major(%alloc: memref<4x4xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0], %cst
          {in_bounds = [false, false],
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<4x4xf16>, vector<16x16xf16>
  // CHECK-COUNT-4: vector.load {{.*}}, vector<8xf16>
  func.return %root : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_col_major
func.func @distribute_transfer_read_col_major(%alloc: memref<32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32xf16>, vector<16x16xf16>
  // CHECK-COUNT-8: vector.load {{.*}}, vector<1xf16>
  func.return %root : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_row_major_with_broadcast
func.func @distribute_transfer_read_row_major_with_broadcast(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  // CHECK-COUNT-4: vector.load {{.*}}, vector<8xf16>
  func.return %root : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_col_major_with_broadcast
func.func @distribute_transfer_read_col_major_with_broadcast(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  // CHECK-COUNT-8: vector.load {{.*}}, vector<1xf16>
  func.return %root : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_row_major_transpose
func.func @distribute_transfer_read_row_major_transpose(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  // CHECK-COUNT-32: vector.load {{.*}}, vector<1xf16>
  func.return %root : vector<16x16xf16>
}

// CHECK-LABEL: @distribute_transfer_read_col_major_transpose
func.func @distribute_transfer_read_col_major_transpose(%a: index, %b: index, %alloc: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %alloc[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  // CHECK-COUNT-2: vector.load {{.*}}, vector<4xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout_row_major = #iree_vector_ext.layout<<[BATCHX, LANEY], [2, 8]>, <[BATCHY, LANEX, VECTORX], [2, 1, 8]>>
#layout_col_major = #iree_vector_ext.layout<<[BATCHX, LANEY, VECTORX], [1, 4, 4]>, <[BATCHY, LANEX], [2, 8]>>

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 * 8 + 8)>

// CHECK-LABEL: @distribute_transfer_write_row_major
func.func @distribute_transfer_write_row_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0]
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<64x64xf16>

  // CHECK-DAG: %[[TIDX:.+]] = gpu.thread_id  x
  // CHECK-DAG: %[[TIDY:.+]] = gpu.thread_id  y
  // CHECK: %[[VECX_IDX:.+]] = affine.apply #[[$MAP0]]()[%[[TIDX]]]
  // CHECK: %[[DIST_SRC_VEC:.+]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xf16> -> vector<2x2x8xf16>
  // CHECK: %[[BATCH_0_0:.+]] = vector.extract %[[DIST_SRC_VEC]][0, 0] : vector<8xf16> from vector<2x2x8xf16>
  // CHECK: vector.store %[[BATCH_0_0]], %{{.*}}[%[[TIDY]], %[[VECX_IDX]]] : memref<64x64xf16>, vector<8xf16>

  // CHECK: %[[NEXT_IDY:.+]] = affine.apply #[[$MAP1]]()[%[[TIDY]]]
  // CHECK: %[[BATCH_1_0:.+]] = vector.extract %[[DIST_SRC_VEC]][1, 0] : vector<8xf16> from vector<2x2x8xf16>
  // CHECK: vector.store %[[BATCH_1_0]], %{{.*}}[%[[NEXT_IDY]], %[[VECX_IDX]]] : memref<64x64xf16>, vector<8xf16>

  // CHECK: %[[NEXT_VECX_IDX:.+]] = affine.apply #[[$MAP2]]()[%[[TIDX]]]
  // CHECK: %[[BATCH_0_1:.+]] = vector.extract %[[DIST_SRC_VEC]][0, 1] : vector<8xf16> from vector<2x2x8xf16>
  // CHECK: vector.store %[[BATCH_0_1]], %{{.*}}[%[[TIDY]], %[[NEXT_VECX_IDX]]] : memref<64x64xf16>, vector<8xf16>

  // CHECK: %[[BATCH_1_1:.+]] = vector.extract %[[DIST_SRC_VEC]][1, 1] : vector<8xf16> from vector<2x2x8xf16>
  // CHECK: vector.store %[[BATCH_1_1]], %{{.*}}[%[[NEXT_IDY]], %[[NEXT_VECX_IDX]]] : memref<64x64xf16>, vector<8xf16>
  func.return
}

// CHECK-LABEL: @distribute_transfer_write_col_major
func.func @distribute_transfer_write_col_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0]
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_operand_0" = #layout_col_major}
                  : vector<16x16xf16>, memref<64x64xf16>
  // CHECK-COUNT-8: vector.store {{.*}}, vector<1xf16>
  func.return
}

// CHECK-LABEL: @distribute_transfer_write_row_major_with_broadcast
func.func @distribute_transfer_write_row_major_with_broadcast(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  // CHECK-COUNT-4: vector.store {{.*}}, vector<8xf16>
  func.return
}

// CHECK-LABEL: @distribute_transfer_write_col_major_with_broadcast
func.func @distribute_transfer_write_col_major_with_broadcast(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_operand_0" = #layout_col_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  // CHECK-COUNT-8: vector.store {{.*}}, vector<1xf16>
  func.return
}

// CHECK-LABEL: @distribute_transfer_write_row_major_transpose
func.func @distribute_transfer_write_row_major_transpose(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  // CHECK-COUNT-32: vector.store {{.*}}, vector<1xf16>
  func.return
}

// CHECK-LABEL: @distribute_transfer_write_col_major_transpose
func.func @distribute_transfer_write_col_major_transpose(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_operand_0" = #layout_col_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>

  // CHECK-COUNT-2: vector.store {{.*}}, vector<4xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
