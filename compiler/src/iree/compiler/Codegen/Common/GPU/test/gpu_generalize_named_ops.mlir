// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-generalize-named-ops))" %s | FileCheck %s

func.func @transpose_matmul(%arg0: tensor<1x4096xf32>, %arg1: tensor<32000x4096xf32>) -> tensor<1x32000xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x32000xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
  %2 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%arg0, %arg1 : tensor<1x4096xf32>, tensor<32000x4096xf32>)
    outs(%1 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
  return %2 : tensor<1x32000xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @transpose_matmul(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x4096xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32000x4096xf32>
// CHECK-SAME:  ) -> tensor<1x32000xf32>
// CHECK:      %[[FILL:.+]] = linalg.fill
// CHECK-SAME: -> tensor<1x32000xf32>
// CHECK:      %[[GEN:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<1x4096xf32>, tensor<32000x4096xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<1x32000xf32>)
// CHECK:      ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:      %[[A0:.+]] = arith.mulf %[[IN]], %[[IN0]] : f32
// CHECK:      %[[A1:.+]] = arith.addf %[[OUT]], %[[A0]] : f32
// CHECK:      linalg.yield %[[A1]] : f32
// CHECK:      return %[[GEN]] : tensor<1x32000xf32>

// -----

func.func @matvec(%arg0: tensor<32000x4096xf32>, %arg1: tensor<4096xf32>) -> tensor<32000xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32000xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32000xf32>) -> tensor<32000xf32>
  %2 = linalg.matvec ins(%arg0, %arg1 : tensor<32000x4096xf32>, tensor<4096xf32>) outs(%1 : tensor<32000xf32>) -> tensor<32000xf32>
  return %2 : tensor<32000xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:  #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @matvec
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[A0:.+]] = arith.mulf %[[IN]], %[[IN0]] : f32
// CHECK:          %[[A1:.+]] = arith.addf %[[OUT]], %[[A0]] : f32
// CHECK:          linalg.yield %[[A1]] : f32

// -----

func.func @vecmat(%arg0: tensor<4096xf32>, %arg1: tensor<4096x32000xf32>) -> tensor<32000xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32000xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32000xf32>) -> tensor<32000xf32>
  %2 = linalg.vecmat ins(%arg0, %arg1 : tensor<4096xf32>, tensor<4096x32000xf32>) outs(%1 : tensor<32000xf32>) -> tensor<32000xf32>
  return %2 : tensor<32000xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG:  #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @vecmat
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[A0:.+]] = arith.mulf %[[IN]], %[[IN0]] : f32
// CHECK:          %[[A1:.+]] = arith.addf %[[OUT]], %[[A0]] : f32
// CHECK:          linalg.yield %[[A1]] : f32

// -----

func.func @transpose_batch_matmul(%arg0: tensor<32x1x128xf16>, %arg1: tensor<32x?x128xf16>, %dim: index) -> tensor<32x1x?xf16> {
  %f0 = arith.constant 0.0 : f16
  %empty = tensor.empty(%dim) : tensor<32x1x?xf16>
  %fill = linalg.fill ins(%f0 : f16) outs(%empty : tensor<32x1x?xf16>) -> tensor<32x1x?xf16>
  %2 = linalg.batch_matmul
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
    ins(%arg0, %arg1 : tensor<32x1x128xf16>, tensor<32x?x128xf16>)
    outs(%fill : tensor<32x1x?xf16>) -> tensor<32x1x?xf16>
  return %2 : tensor<32x1x?xf16>
}

//       CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @transpose_batch_matmul
//       CHECK:  linalg.generic
//  CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//       CHECK:  ^bb0(%[[A:.+]]: f16, %[[B:.+]]: f16, %[[OUT:.+]]: f16):
//       CHECK:    %[[MUL:.+]] = arith.mulf %[[A]], %[[B]] : f16
//       CHECK:    %[[ADD:.+]] = arith.addf %[[OUT]], %[[MUL]] : f16
//       CHECK:    linalg.yield %[[ADD]] : f16

// -----

func.func @lowering_config(%arg0: tensor<512x128xf16>, %arg1: tensor<512x128xf16>) -> tensor<512x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<512x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %2 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[16, 16, 16]]>}
    ins(%arg0, %arg1 : tensor<512x128xf16>, tensor<512x128xf16>)
    outs(%1 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %2 : tensor<512x512xf32>
}

//               CHECK: #[[$CONFIG:.+]] = #iree_codegen.lowering_config
// CHECK-SAME{LITERAL}: <tile_sizes = [[16, 16, 16]]>

// CHECK-LABEL: func.func @lowering_config
//       CHECK:   linalg.generic
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

func.func @transpose_op(%arg0: tensor<16x32xf16>) -> tensor<32x16xf16> {
  %empty = tensor.empty() : tensor<32x16xf16>
  %transpose = linalg.transpose
      ins(%arg0 : tensor<16x32xf16>)
      outs(%empty : tensor<32x16xf16>)
      permutation = [1, 0]
  return %transpose : tensor<32x16xf16>
}

// CHECK-LABEL: func.func @transpose_op
//       CHECK:   linalg.generic

// -----

func.func @reduce_op(%arg0: tensor<32x128xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
  %reduced = linalg.reduce ins(%arg0 : tensor<32x128xf32>) outs(%arg1 : tensor<32xf32>) dimensions = [1]
    (%in: f32, %out: f32) {
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    }
  return %reduced : tensor<32xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @reduce_op
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP1]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[SUM:.+]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[SUM]] : f32

// -----

func.func @arg_compare_argmax(%input: tensor<4x128xf32>,
                              %out_val: tensor<4xf32>,
                              %out_idx: tensor<4xi32>)
    -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xf32>)
      outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @arg_compare_argmax
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<4x128xf32>
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9_]+]]: tensor<4xi32>
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK-SAME:   ins(%[[INPUT]] : tensor<4x128xf32>)
// CHECK-SAME:   outs(%[[OUT_VAL]], %[[OUT_IDX]] : tensor<4xf32>, tensor<4xi32>)
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[ACC_VAL:.+]]: f32, %[[ACC_IDX:.+]]: i32):
// CHECK:          %[[IDX:.+]] = linalg.index 1 : index
// CHECK:          %[[IDX_CAST:.+]] = arith.index_cast %[[IDX]] : index to i32
// CHECK:          %[[CMP:.+]] = arith.cmpf ogt, %[[IN]], %[[ACC_VAL]] : f32
// CHECK:          %[[NEW_VAL:.+]] = arith.select %[[CMP]], %[[IN]], %[[ACC_VAL]] : f32
// CHECK:          %[[NEW_IDX:.+]] = arith.select %[[CMP]], %[[IDX_CAST]], %[[ACC_IDX]] : i32
// CHECK:          linalg.yield %[[NEW_VAL]], %[[NEW_IDX]] : f32, i32

// -----

func.func @arg_compare_with_index_base(%input: tensor<4x128xf32>,
                                        %out_val: tensor<4xf32>,
                                        %out_idx: tensor<4xi32>,
                                        %index_base: index)
    -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xf32>)
      outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>)
      index_base(%index_base : index) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @arg_compare_with_index_base
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<4x128xf32>
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9_]+]]: tensor<4xi32>
// CHECK-SAME:    %[[INDEX_BASE:[a-zA-Z0-9_]+]]: index
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[ACC_VAL:.+]]: f32, %[[ACC_IDX:.+]]: i32):
// CHECK:          %[[IDX:.+]] = linalg.index 1 : index
// CHECK:          %[[IDX_OFFSET:.+]] = arith.addi %[[IDX]], %[[INDEX_BASE]] : index
// CHECK:          %[[IDX_CAST:.+]] = arith.index_cast %[[IDX_OFFSET]] : index to i32
// CHECK:          %[[CMP:.+]] = arith.cmpf ogt, %[[IN]], %[[ACC_VAL]] : f32
// CHECK:          %[[NEW_VAL:.+]] = arith.select %[[CMP]], %[[IN]], %[[ACC_VAL]] : f32
// CHECK:          %[[NEW_IDX:.+]] = arith.select %[[CMP]], %[[IDX_CAST]], %[[ACC_IDX]] : i32
// CHECK:          linalg.yield %[[NEW_VAL]], %[[NEW_IDX]] : f32, i32

// -----

func.func @arg_compare_index_output(%input: tensor<4x128xf32>,
                                    %out_val: tensor<4xf32>,
                                    %out_idx: tensor<4xindex>)
    -> (tensor<4xf32>, tensor<4xindex>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xf32>)
      outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xindex>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xindex>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xindex>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @arg_compare_index_output
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<4x128xf32>
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9_]+]]: tensor<4xindex>
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK-SAME:   outs(%[[OUT_VAL]], %[[OUT_IDX]] : tensor<4xf32>, tensor<4xindex>)
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[ACC_VAL:.+]]: f32, %[[ACC_IDX:.+]]: index):
// CHECK:          %[[IDX:.+]] = linalg.index 1 : index
// CHECK-NOT:      arith.index_cast
// CHECK:          %[[CMP:.+]] = arith.cmpf ogt, %[[IN]], %[[ACC_VAL]] : f32
// CHECK:          %[[NEW_VAL:.+]] = arith.select %[[CMP]], %[[IN]], %[[ACC_VAL]] : f32
// CHECK:          %[[NEW_IDX:.+]] = arith.select %[[CMP]], %[[IDX]], %[[ACC_IDX]] : index
// CHECK:          linalg.yield %[[NEW_VAL]], %[[NEW_IDX]] : f32, index
