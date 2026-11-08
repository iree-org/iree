// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

func.func @sort_drop_unused_results(%arg0 : tensor<?x10xf32>,
    %arg1 : tensor<?x10xi64>) -> tensor<?x10xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x10xf32>
  %d1 = tensor.dim %arg1, %c0 : tensor<?x10xi64>
  %empty0 = tensor.empty(%d0) : tensor<?x10xf32>
  %empty1 = tensor.empty(%d1) : tensor<?x10xi64>
  %0:2 = iree_linalg_ext.sort dimension(1) ins(%arg0, %arg1 : tensor<?x10xf32>,
      tensor<?x10xi64>) outs(%empty0, %empty1 : tensor<?x10xf32>,
      tensor<?x10xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %42 = arith.cmpf oge, %arg2, %arg3 : f32
    iree_linalg_ext.yield %42 : i1
  } -> tensor<?x10xf32>, tensor<?x10xi64>
  return %0#0 : tensor<?x10xf32>
}
// CHECK-LABEL: func.func @sort_drop_unused_results
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x10xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?x10xi64>
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x10xf32>
//       CHECK:   %[[EMPTY0:.+]] = tensor.empty(%[[D0]]) : tensor<?x10xf32>
//       CHECK:   %[[SORT:.+]] = iree_linalg_ext.sort
//       CHECK:       dimension(1)
//       CHECK:       ins(%[[ARG0]] : tensor<?x10xf32>) outs(%[[EMPTY0]] : tensor<?x10xf32>)
//       CHECK:   return %[[SORT]] : tensor<?x10xf32>

// -----

// A stable sort with an operand that holds the positions along the sort
// dimension: the comparator orders the elements it does not order itself by
// position, and the sort does not have to be stable anymore.
func.func @sort_fold_stability_into_comparator(%arg0: tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %empty_pos = tensor.empty() : tensor<3x4xi64>
  %positions = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%empty_pos : tensor<3x4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 1 : index
    %i64 = arith.index_cast %i : index to i64
    linalg.yield %i64 : i64
  } -> tensor<3x4xi64>
  %empty = tensor.empty() : tensor<3x4xf32>
  %0:2 = iree_linalg_ext.sort dimension(1) is_stable(true)
      ins(%arg0, %positions : tensor<3x4xf32>, tensor<3x4xi64>)
      outs(%empty, %empty_pos : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf oge, %arg2, %arg3 : f32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}
// CHECK-LABEL: func.func @sort_fold_stability_into_comparator
//       CHECK:   %[[POSITIONS:.+]] = linalg.generic
//       CHECK:   iree_linalg_ext.sort
//   CHECK-NOT:       is_stable
//  CHECK-SAME:       dimension(1)
//  CHECK-SAME:       ins(%{{.+}}, %[[POSITIONS]] : tensor<3x4xf32>, tensor<3x4xi64>)
//       CHECK:   ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32, %[[LHS_POS:.+]]: i64, %[[RHS_POS:.+]]: i64):
//       CHECK:     %[[BEFORE:.+]] = arith.cmpf oge, %[[LHS]], %[[RHS]] : f32
//       CHECK:     %[[AFTER:.+]] = arith.cmpf oge, %[[RHS]], %[[LHS]] : f32
//       CHECK:     %[[ORDERED:.+]] = arith.xori %[[BEFORE]], %[[AFTER]] : i1
//       CHECK:     %[[LOWER_POS:.+]] = arith.cmpi slt, %[[LHS_POS]], %[[RHS_POS]] : i64
//       CHECK:     %[[RESULT:.+]] = arith.select %[[ORDERED]], %[[BEFORE]], %[[LOWER_POS]] : i1
//       CHECK:     iree_linalg_ext.yield %[[RESULT]] : i1

// -----

// Positions along another dimension than the sorted one do not tell the input
// order of the elements of a sorted slice. The sort stays stable.
func.func @sort_stable_positions_of_other_dimension(%arg0: tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %empty_pos = tensor.empty() : tensor<3x4xi64>
  %positions = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%empty_pos : tensor<3x4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 0 : index
    %i64 = arith.index_cast %i : index to i64
    linalg.yield %i64 : i64
  } -> tensor<3x4xi64>
  %empty = tensor.empty() : tensor<3x4xf32>
  %0:2 = iree_linalg_ext.sort dimension(1) is_stable(true)
      ins(%arg0, %positions : tensor<3x4xf32>, tensor<3x4xi64>)
      outs(%empty, %empty_pos : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf oge, %arg2, %arg3 : f32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}
// CHECK-LABEL: func.func @sort_stable_positions_of_other_dimension
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:       dimension(1) is_stable(true)
//       CHECK:   ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32, %{{.+}}: i64, %{{.+}}: i64):
//  CHECK-NEXT:     %[[BEFORE:.+]] = arith.cmpf oge, %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:     iree_linalg_ext.yield %[[BEFORE]] : i1

// -----

// The sorted positions are not used. The operand is dropped instead of being
// kept for the comparator, and the sort of the remaining operand stays stable.
func.func @sort_stable_unused_positions(%arg0: tensor<3x4xf32>)
    -> tensor<3x4xf32> {
  %empty_pos = tensor.empty() : tensor<3x4xi64>
  %positions = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%empty_pos : tensor<3x4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 1 : index
    %i64 = arith.index_cast %i : index to i64
    linalg.yield %i64 : i64
  } -> tensor<3x4xi64>
  %empty = tensor.empty() : tensor<3x4xf32>
  %0:2 = iree_linalg_ext.sort dimension(1) is_stable(true)
      ins(%arg0, %positions : tensor<3x4xf32>, tensor<3x4xi64>)
      outs(%empty, %empty_pos : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf oge, %arg2, %arg3 : f32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0 : tensor<3x4xf32>
}
// CHECK-LABEL: func.func @sort_stable_unused_positions
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<3x4xf32>
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:       dimension(1) is_stable(true)
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<3x4xf32>)
//       CHECK:   ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32):
//  CHECK-NEXT:     %[[BEFORE:.+]] = arith.cmpf oge, %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:     iree_linalg_ext.yield %[[BEFORE]] : i1

// -----

// A sort that is not stable is left alone.
func.func @sort_unstable_with_positions(%arg0: tensor<3x4xf32>)
    -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %empty_pos = tensor.empty() : tensor<3x4xi64>
  %positions = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%empty_pos : tensor<3x4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 1 : index
    %i64 = arith.index_cast %i : index to i64
    linalg.yield %i64 : i64
  } -> tensor<3x4xi64>
  %empty = tensor.empty() : tensor<3x4xf32>
  %0:2 = iree_linalg_ext.sort dimension(1)
      ins(%arg0, %positions : tensor<3x4xf32>, tensor<3x4xi64>)
      outs(%empty, %empty_pos : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf oge, %arg2, %arg3 : f32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}
// CHECK-LABEL: func.func @sort_unstable_with_positions
//       CHECK:   iree_linalg_ext.sort
//       CHECK:   ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32, %{{.+}}: i64, %{{.+}}: i64):
//  CHECK-NEXT:     %[[BEFORE:.+]] = arith.cmpf oge, %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:     iree_linalg_ext.yield %[[BEFORE]] : i1

// -----

// Integers that an integer comparison does not order are the same value, so
// their order cannot be observed and the sort does not have to be stable.
func.func @sort_drop_unobservable_stability(%arg0: tensor<128xi32>)
    -> tensor<128xi32> {
  %empty = tensor.empty() : tensor<128xi32>
  %0 = iree_linalg_ext.sort dimension(0) is_stable(true)
      ins(%arg0 : tensor<128xi32>) outs(%empty : tensor<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.cmpi sle, %arg1, %arg2 : i32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<128xi32>
  return %0 : tensor<128xi32>
}
// CHECK-LABEL: func.func @sort_drop_unobservable_stability
//       CHECK:   iree_linalg_ext.sort
//   CHECK-NOT:       is_stable
//  CHECK-SAME:       dimension(0)
//  CHECK-SAME:       ins(

// -----

// Floats that compare equal can differ (0.0 and -0.0), and a comparator that
// does more than comparing its arguments can leave different elements
// unordered. These sorts stay stable.
func.func @sort_keep_observable_stability(%arg0: tensor<128xf32>,
    %arg1: tensor<128xi32>) -> (tensor<128xf32>, tensor<128xi32>) {
  %empty0 = tensor.empty() : tensor<128xf32>
  %0 = iree_linalg_ext.sort dimension(0) is_stable(true)
      ins(%arg0 : tensor<128xf32>) outs(%empty0 : tensor<128xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = arith.cmpf ole, %arg2, %arg3 : f32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<128xf32>
  %empty1 = tensor.empty() : tensor<128xi32>
  %2 = iree_linalg_ext.sort dimension(0) is_stable(true)
      ins(%arg1 : tensor<128xi32>) outs(%empty1 : tensor<128xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %c4 = arith.constant 4 : i32
    %lhs = arith.shrsi %arg2, %c4 : i32
    %rhs = arith.shrsi %arg3, %c4 : i32
    %3 = arith.cmpi sle, %lhs, %rhs : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<128xi32>
  return %0, %2 : tensor<128xf32>, tensor<128xi32>
}
// CHECK-LABEL: func.func @sort_keep_observable_stability
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:       dimension(0) is_stable(true)
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:       dimension(0) is_stable(true)

// -----

func.func @gather_to_extract_slice_expand(%source : tensor<1024x128xi32>, %indices : tensor<1x1xi32>) -> (tensor<1x1x128xi32>) {
  %empty = tensor.empty() : tensor<1x1x128xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
    ins(%source, %indices : tensor<1024x128xi32>, tensor<1x1xi32>)
    outs(%empty: tensor<1x1x128xi32>) -> tensor<1x1x128xi32>
  return %result : tensor<1x1x128xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_expand
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[IDX:.+]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]]]
//       CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST]], 0] [1, 128] [1, 1]
//  CHECK-SAME:     tensor<1024x128xi32> to tensor<1x128xi32>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]]
//  CHECK-SAME:     tensor<1x128xi32> into tensor<1x1x128xi32>

// -----

func.func @gather_to_extract_slice_no_reshape(%source : tensor<1024x128xi32>, %indices : tensor<1xi32>) -> (tensor<1x128xi32>) {
  %empty = tensor.empty() : tensor<1x128xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
    ins(%source, %indices : tensor<1024x128xi32>, tensor<1xi32>)
    outs(%empty: tensor<1x128xi32>) -> tensor<1x128xi32>
  return %result : tensor<1x128xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_no_reshape
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[IDX:.+]] = tensor.extract %[[ARG1]][%[[C0]]]
//       CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST]], 0] [1, 128] [1, 1]
//  CHECK-SAME:     tensor<1024x128xi32> to tensor<1x128xi32>

// -----

func.func @gather_to_extract_slice_perm(%source : tensor<10x1024x128xi32>, %indices : tensor<1x1x2xi32>) -> (tensor<1x1x128xi32>) {
  %empty = tensor.empty() : tensor<1x1x128xi32>
  %result = iree_linalg_ext.gather dimension_map = [1, 0]
    ins(%source, %indices : tensor<10x1024x128xi32>, tensor<1x1x2xi32>)
    outs(%empty: tensor<1x1x128xi32>) -> tensor<1x1x128xi32>
  return %result : tensor<1x1x128xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_perm
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[IDX0:.+]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]], %[[C0]]]
//   CHECK-DAG:   %[[IDX1:.+]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]], %[[C1]]]
//   CHECK-DAG:   %[[CAST0:.+]] = arith.index_cast %[[IDX0]]
//   CHECK-DAG:   %[[CAST1:.+]] = arith.index_cast %[[IDX1]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST1]], %[[CAST0]], 0] [1, 1, 128] [1, 1, 1]
//  CHECK-SAME:     tensor<10x1024x128xi32> to tensor<1x1x128xi32>

// -----

func.func @gather_to_extract_slice_full_collapse(%source : tensor<2x2x1xi32>, %indices : tensor<2xi32>) -> (tensor<1xi32>) {
  %empty = tensor.empty() : tensor<1xi32>
  %result = iree_linalg_ext.gather dimension_map = [0, 1]
    ins(%source, %indices : tensor<2x2x1xi32>, tensor<2xi32>)
    outs(%empty: tensor<1xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_full_collapse
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[IDX0:.+]] = tensor.extract %[[ARG1]][%[[C0]]]
//   CHECK-DAG:   %[[IDX1:.+]] = tensor.extract %[[ARG1]][%[[C1]]]
//   CHECK-DAG:   %[[CAST0:.+]] = arith.index_cast %[[IDX0]]
//   CHECK-DAG:   %[[CAST1:.+]] = arith.index_cast %[[IDX1]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST0]], %[[CAST1]], 0] [1, 1, 1] [1, 1, 1]
//  CHECK-SAME:     tensor<2x2x1xi32> to tensor<1x1x1xi32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SLICE]]
//  CHECK-SAME:     tensor<1x1x1xi32> into tensor<1xi32>

// -----

func.func @gather_to_extract_slice_partial_collapse(%source : tensor<2x2x100x100xi32>, %indices : tensor<2xi32>) -> (tensor<100x100xi32>) {
  %empty = tensor.empty() : tensor<100x100xi32>
  %result = iree_linalg_ext.gather dimension_map = [1, 0]
    ins(%source, %indices : tensor<2x2x100x100xi32>, tensor<2xi32>)
    outs(%empty: tensor<100x100xi32>) -> tensor<100x100xi32>
  return %result : tensor<100x100xi32>
}
// CHECK-LABEL: @gather_to_extract_slice_partial_collapse
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[IDX0:.+]] = tensor.extract %[[ARG1]][%[[C0]]]
//   CHECK-DAG:   %[[IDX1:.+]] = tensor.extract %[[ARG1]][%[[C1]]]
//   CHECK-DAG:   %[[CAST0:.+]] = arith.index_cast %[[IDX0]]
//   CHECK-DAG:   %[[CAST1:.+]] = arith.index_cast %[[IDX1]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[CAST1]], %[[CAST0]], 0, 0] [1, 1, 100, 100] [1, 1, 1, 1]
//  CHECK-SAME:     tensor<2x2x100x100xi32> to tensor<1x1x100x100xi32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SLICE]]
//  CHECK-SAME:     tensor<1x1x100x100xi32> into tensor<100x100xi32>

// -----

func.func public @staticize_attention_from_operand(%arg0: tensor<?x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16) -> tensor<20x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = iree_linalg_ext.attention <indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]> ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16) outs(%0 : tensor<20x4096x64xf16>) {
  ^bb0(%arg4: f16):
    iree_linalg_ext.yield %arg4 : f16
  } -> tensor<20x4096x64xf16>
  return %1 : tensor<20x4096x64xf16>
}
//CHECK-LABEL: func public @staticize_attention_from_operand(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//      CHECK:   %[[CAST:.+]] = tensor.cast %[[ARG0]]
// CHECK-SAME:     : tensor<?x4096x16xf16> to tensor<20x4096x16xf16>
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       ins(%[[CAST]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//      CHECK:   return %[[ATTENTION]]

// -----

func.func public @staticize_attention_from_cast(%arg0: tensor<?x4096x16xf16>, %arg1: tensor<20x?x16xf16>, %arg2: tensor<20x?x64xf16>, %arg3: f16) -> tensor<20x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %cast0 = tensor.cast %arg1 : tensor<20x?x16xf16> to tensor<?x?x16xf16>
  %cast1 = tensor.cast %arg2 : tensor<20x?x64xf16> to tensor<?x?x64xf16>
  %1 = iree_linalg_ext.attention <indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]> ins(%arg0, %cast0, %cast1, %arg3 : tensor<?x4096x16xf16>, tensor<?x?x16xf16>, tensor<?x?x64xf16>, f16) outs(%0 : tensor<20x4096x64xf16>) {
  ^bb0(%arg4: f16):
    iree_linalg_ext.yield %arg4 : f16
  } -> tensor<20x4096x64xf16>
  return %1 : tensor<20x4096x64xf16>
}
//CHECK-LABEL: func public @staticize_attention_from_cast(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x?x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x?x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//      CHECK:   %[[CAST:.+]] = tensor.cast %[[ARG0]]
// CHECK-SAME:     : tensor<?x4096x16xf16> to tensor<20x4096x16xf16>
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       ins(%[[CAST]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//      CHECK:   return %[[ATTENTION]]

// -----

func.func public @staticize_online_attention_from_cast(%arg0: tensor<?x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16) -> tensor<20x4096x64xf16> {
  %0 = tensor.empty() : tensor<20x4096x64xf16>
  %1 = tensor.empty() : tensor<20x4096xf16>
  %cast0 = tensor.cast %arg1 : tensor<20x1024x16xf16> to tensor<?x1024x16xf16>
  %cast1 = tensor.cast %arg2 : tensor<20x1024x64xf16> to tensor<?x1024x64xf16>
  %2:3 = iree_linalg_ext.online_attention <indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>]> ins(%arg0, %cast0, %cast1, %arg3 : tensor<?x4096x16xf16>, tensor<?x1024x16xf16>, tensor<?x1024x64xf16>, f16) outs(%0, %1, %1 : tensor<20x4096x64xf16>, tensor<20x4096xf16>, tensor<20x4096xf16>) {
  ^bb0(%arg4: f16):
    iree_linalg_ext.yield %arg4 : f16
  } -> tensor<20x4096x64xf16>, tensor<20x4096xf16>, tensor<20x4096xf16>
  return %2#0 : tensor<20x4096x64xf16>
}
//CHECK-LABEL: func public @staticize_online_attention_from_cast(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//      CHECK:   %[[CAST:.+]] = tensor.cast %[[ARG0]]
// CHECK-SAME:     : tensor<?x4096x16xf16> to tensor<20x4096x16xf16>
//      CHECK:   %[[ATTENTION:.+]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:       ins(%[[CAST]], %[[ARG1]], %[[ARG2]], %[[ARG3]] :
//      CHECK:   return %[[ATTENTION]]#0

// -----

func.func public @convert_identity_map_store_into_copy(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %true = arith.constant true
  %map_store = iree_linalg_ext.map_store %arg0 into %arg1 {
  ^bb0(%arg2: index, %arg3: index):
    iree_linalg_ext.yield %arg2, %arg3, %true : index, index, i1
  } : tensor<?x?xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
  return %map_store : tensor<?x?xf32>
}
//CHECK-LABEL: func public @convert_identity_map_store_into_copy(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-NOT:   iree_linalg_ext.map_store
//      CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[ARG0]]{{.*}} outs(%[[ARG1]]
//      CHECK:   return %[[COPY]]

// -----

func.func public @convert_identity_map_load_into_copy(
    %arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>
) -> tensor<16x32xf32> {
  %cst = arith.constant 0.0 : f32
  %map_load = iree_linalg_ext.map_load %arg0 into %arg1 {
  ^bb0(%arg2: index, %arg3: index):
    iree_linalg_ext.yield %arg2, %arg3, %cst : index, index, f32
  } : tensor<16x32xf32> into tensor<16x32xf32> -> tensor<16x32xf32>
  return %map_load : tensor<16x32xf32>
}
//CHECK-LABEL: func public @convert_identity_map_load_into_copy(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x32xf32>
//  CHECK-NOT:   iree_linalg_ext.map_load
//      CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[ARG0]]{{.*}} outs(%[[ARG1]]
//      CHECK:   return %[[COPY]]

// -----

// Verify that identity map_load with dynamic shapes is not converted to copy.
func.func public @no_convert_dynamic_identity_map_load_into_copy(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %map_load = iree_linalg_ext.map_load %arg0 into %arg1 {
  ^bb0(%arg2: index, %arg3: index):
    iree_linalg_ext.yield %arg2, %arg3, %cst : index, index, f32
  } : tensor<?x?xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
  return %map_load : tensor<?x?xf32>
}
//CHECK-LABEL: func public @no_convert_dynamic_identity_map_load_into_copy(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//      CHECK:   iree_linalg_ext.map_load
//  CHECK-NOT:   linalg.copy

// -----

// Test: fold input tensor.pad into im2col padding attributes.

func.func @fold_input_pad_into_im2col(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1296x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1296x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      ins(%padded : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1296x5760xf32>) -> tensor<2x1296x5760xf32>
  return %result : tensor<2x1296x5760xf32>
}
// CHECK-LABEL: func.func @fold_input_pad_into_im2col(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1296x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%[[CST]] : f32)
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1296x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold output tensor.pad into im2col by expanding the output tensor.

func.func @fold_output_pad_into_im2col(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1040x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x1024x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%cst : f32)
      ins(%arg0 : tensor<2x34x34x640xf32>)
      outs(%empty : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  %padded = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1024x5760xf32> to tensor<2x1040x5760xf32>
  return %padded : tensor<2x1040x5760xf32>
}
// CHECK-LABEL: func.func @fold_output_pad_into_im2col(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1040x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1040x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: both input+output pads with incompatible pad values — neither folds.

func.func @no_fold_both_pads_incompatible(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1040x5760xf32> {
  %cst_pad = arith.constant 1.0 : f32
  %cst_im2col = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst_pad : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1024x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 0, 0, 0] input_pad_high = [0, 0, 0, 0] pad_value(%cst_im2col : f32)
      ins(%padded : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  %padded_output = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst_pad : f32
  } : tensor<2x1024x5760xf32> to tensor<2x1040x5760xf32>
  return %padded_output : tensor<2x1040x5760xf32>
}
// CHECK-LABEL: func.func @no_fold_both_pads_incompatible(
//       CHECK:   tensor.pad
//       CHECK:   iree_linalg_ext.im2col
//       CHECK:   tensor.pad

// -----

// Test: both input+output pads with compatible pad values — both fold.

func.func @fold_both_pads_compatible(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1312x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %padded_input = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1296x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      ins(%padded_input : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1296x5760xf32>) -> tensor<2x1296x5760xf32>
  %padded_output = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1296x5760xf32> to tensor<2x1312x5760xf32>
  return %padded_output : tensor<2x1312x5760xf32>
}
// CHECK-LABEL: func.func @fold_both_pads_compatible(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1312x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
//  CHECK-SAME:     output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
//  CHECK-SAME:     pad_value(%[[CST]] : f32)
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1312x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold input tensor.pad into im2col that already has input padding
// (composing by adding element-wise).

func.func @fold_input_pad_compose(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1296x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
      tensor.yield %cst : f32
  } : tensor<2x34x34x640xf32> to tensor<2x36x36x640xf32>
  %empty = tensor.empty() : tensor<2x1296x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%cst : f32)
      ins(%padded : tensor<2x36x36x640xf32>)
      outs(%empty : tensor<2x1296x5760xf32>) -> tensor<2x1296x5760xf32>
  return %result : tensor<2x1296x5760xf32>
}
// CHECK-LABEL: func.func @fold_input_pad_compose(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     input_pad_low = [0, 2, 2, 0] input_pad_high = [0, 2, 2, 0] pad_value(%[[CST]] : f32)
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold output tensor.pad with non-zero low padding into im2col.

func.func @fold_output_pad_low(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1040x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x1024x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0] pad_value(%cst : f32)
      ins(%arg0 : tensor<2x34x34x640xf32>)
      outs(%empty : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  %padded = tensor.pad %result low[0, 16, 0] high[0, 0, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1024x5760xf32> to tensor<2x1040x5760xf32>
  return %padded : tensor<2x1040x5760xf32>
}
// CHECK-LABEL: func.func @fold_output_pad_low(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1040x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     output_pad_low = [0, 16, 0] output_pad_high = [0, 0, 0]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1040x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

// Test: fold output tensor.pad composing with existing output padding.

func.func @fold_output_pad_compose(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1056x5760xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x1040x5760xf32>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      offsets = [0, 0, 0] output_sizes = [[2], [32, 32], [3, 3, 640]]
      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
      input_k_perm = [0, 1, 2]
      output_perm = [0, 1, 2]
      output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
      pad_value(%cst : f32)
      ins(%arg0 : tensor<2x34x34x640xf32>)
      outs(%empty : tensor<2x1040x5760xf32>) -> tensor<2x1040x5760xf32>
  %padded = tensor.pad %result low[0, 0, 0] high[0, 16, 0] {
    ^bb0(%b0: index, %b1: index, %b2: index):
      tensor.yield %cst : f32
  } : tensor<2x1040x5760xf32> to tensor<2x1056x5760xf32>
  return %padded : tensor<2x1056x5760xf32>
}
// CHECK-LABEL: func.func @fold_output_pad_compose(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x34x34x640xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1056x5760xf32>
//       CHECK:   %[[IM2COL:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     output_pad_low = [0, 0, 0] output_pad_high = [0, 32, 0]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<2x1056x5760xf32>)
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[IM2COL]]

// -----

func.func @dequantize_affine_drop_zero_zero_point(%q: tensor<32x64xi8>,
    %s: tensor<32xf32>, %init: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %zp = arith.constant dense<0> : tensor<32xi64>
  %0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q, %s, %zp : tensor<32x64xi8>, tensor<32xf32>, tensor<32xi64>)
      outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_drop_zero_zero_point
//   CHECK-NOT:   arith.constant dense<0>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     indexing_maps = [#{{[^,]+}}, #{{[^,]+}}, #{{[^],]+}}]
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<32x64xi8>, tensor<32xf32>)

// -----

// Zeros that are not spelled as a splat, and a zp_unsigned flag that has to go
// with the operand it describes.
func.func @dequantize_affine_drop_nonsplat_zeros(%q: tensor<4x2xi8>,
    %s: tensor<4xf32>, %init: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %zp = arith.constant dense<[0, 0, 0, 0]> : tensor<4xi32>
  %0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>], zp_unsigned>
      ins(%q, %s, %zp : tensor<4x2xi8>, tensor<4xf32>, tensor<4xi32>)
      outs(%init : tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_drop_nonsplat_zeros
//   CHECK-NOT:   zp_unsigned
//       CHECK:   ins(%{{.+}}, %{{.+}} : tensor<4x2xi8>, tensor<4xf32>)

// -----

func.func @quantize_affine_drop_zero_scalar_zero_point(%x: tensor<8xf32>, %s: f32,
    %init: tensor<8xi8>) -> tensor<8xi8> {
  %zp = arith.constant 0 : i64
  %0 = iree_linalg_ext.quantize_affine
      <indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>,
                        affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
       quant_min = 0, quant_max = 255, storage_unsigned,
       zp_unsigned>
      ins(%x, %s, %zp : tensor<8xf32>, f32, i64)
      outs(%init : tensor<8xi8>) -> tensor<8xi8>
  return %0 : tensor<8xi8>
}
// CHECK-LABEL: func.func @quantize_affine_drop_zero_scalar_zero_point
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     quant_min = 0
//  CHECK-SAME:     quant_max = 255
//  CHECK-SAME:     storage_unsigned
//   CHECK-NOT:     zp_unsigned
//       CHECK:     ins(%{{.+}}, %{{.+}} : tensor<8xf32>, f32)

// A nonzero element keeps the operand.
func.func @dequantize_affine_keep_nonzero_zero_point(%q: tensor<4x2xi8>,
    %s: tensor<4xf32>, %init: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %zp = arith.constant dense<[0, 3, 0, 0]> : tensor<4xi32>
  %0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q, %s, %zp : tensor<4x2xi8>, tensor<4xf32>, tensor<4xi32>)
      outs(%init : tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_keep_nonzero_zero_point
//       CHECK:   arith.constant dense<[0, 3, 0, 0]>
//       CHECK:   ins(%{{.+}}, %{{.+}}, %{{.+}} :

// A dynamic zero point cannot be proven zero.
func.func @dequantize_affine_keep_dynamic_zero_point(%q: tensor<4x2xi8>,
    %s: tensor<4xf32>, %zp: tensor<4xi32>, %init: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q, %s, %zp : tensor<4x2xi8>, tensor<4xf32>, tensor<4xi32>)
      outs(%init : tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_keep_dynamic_zero_point
//       CHECK:   ins(%{{.+}}, %{{.+}}, %{{.+}} :

func.func @dequantize_affine_drop_zero_resource_zero_point(%q: tensor<4x2xi8>,
    %s: tensor<4xf32>, %init: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %zp = arith.constant dense_resource<zero_zero_points> : tensor<4xi64>
  %0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q, %s, %zp : tensor<4x2xi8>, tensor<4xf32>, tensor<4xi64>)
      outs(%init : tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_drop_zero_resource_zero_point
//       CHECK:   ins(%{{.+}}, %{{.+}} : tensor<4x2xi8>, tensor<4xf32>)

// A resource blob with a nonzero byte keeps the operand.
func.func @dequantize_affine_keep_nonzero_resource_zero_point(%q: tensor<4x2xi8>,
    %s: tensor<4xf32>, %init: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %zp = arith.constant dense_resource<nonzero_zero_points> : tensor<4xi64>
  %0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q, %s, %zp : tensor<4x2xi8>, tensor<4xf32>, tensor<4xi64>)
      outs(%init : tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_keep_nonzero_resource_zero_point
//       CHECK:   ins(%{{.+}}, %{{.+}}, %{{.+}} :

{-#
  dialect_resources: {
    builtin: {
      zero_zero_points: "0x080000000000000000000000000000000000000000000000000000000000000000000000",
      nonzero_zero_points: "0x080000000000000000000000000000000000000000000000000000000000000000000500"
    }
  }
#-}

// -----

// CHECK-LABEL: @general_scatter_preserve_effect
// CHECK: iree_linalg_ext.general_scatter
// CHECK: memref.store
// CHECK: return
func.func @general_scatter_preserve_effect(%updates: tensor<2xi32>, %indices: tensor<2x1xi64>, %original: tensor<4xi32>, %buffer: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %result = iree_linalg_ext.general_scatter <index_vector_dim = 1, update_window_dims = [], inserted_window_dims = [0], input_batching_dims = [], scatter_indices_batching_dims = [], scatter_dims_to_operand_dims = [0]>
    ins(%updates : tensor<2xi32>) indices(%indices : tensor<2x1xi64>)
    outs(%original : tensor<4xi32>) {
  ^bb0(%current: i32, %update: i32):
    memref.store %update, %buffer[%c0] : memref<1xi32>
    iree_linalg_ext.yield %update : i32
  } -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: @general_scatter_drop_unused_pure
// CHECK-NOT: iree_linalg_ext.general_scatter
// CHECK: return
func.func @general_scatter_drop_unused_pure(%updates: tensor<2xi32>, %indices: tensor<2x1xi64>, %original: tensor<4xi32>, %buffer: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %result = iree_linalg_ext.general_scatter <index_vector_dim = 1, update_window_dims = [], inserted_window_dims = [0], input_batching_dims = [], scatter_indices_batching_dims = [], scatter_dims_to_operand_dims = [0]>
    ins(%updates : tensor<2xi32>) indices(%indices : tensor<2x1xi64>)
    outs(%original : tensor<4xi32>) {
  ^bb0(%current: i32, %update: i32):
    iree_linalg_ext.yield %update : i32
  } -> tensor<4xi32>
  return
}

// -----

// With no update elements nothing is scattered.

// CHECK-LABEL: func.func @scatter_no_updates(
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]: tensor<0x3xf32>, %[[INDICES:[a-zA-Z0-9]+]]: tensor<0x1xi32>, %[[ORIGINAL:[a-zA-Z0-9]+]]: tensor<5x3xf32>
// CHECK-NOT:     iree_linalg_ext.scatter
// CHECK:         return %[[ORIGINAL]]
func.func @scatter_no_updates(%updates: tensor<0x3xf32>, %indices: tensor<0x1xi32>, %original: tensor<5x3xf32>) -> tensor<5x3xf32> {
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : tensor<0x3xf32>, tensor<0x1xi32>)
    outs(%original : tensor<5x3xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %1 = arith.addf %arg0, %arg1 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}

// -----

// CHECK-LABEL: func.func @scatter_with_updates(
// CHECK:         iree_linalg_ext.scatter
func.func @scatter_with_updates(%updates: tensor<2x3xf32>, %indices: tensor<2x1xi32>, %original: tensor<5x3xf32>) -> tensor<5x3xf32> {
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : tensor<2x3xf32>, tensor<2x1xi32>)
    outs(%original : tensor<5x3xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %1 = arith.addf %arg0, %arg1 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}

// -----

// CHECK-LABEL: func.func @general_scatter_no_updates(
// CHECK-SAME:    %[[ORIGINAL0:[a-zA-Z0-9]+]]: tensor<6x4xi32>, %[[ORIGINAL1:[a-zA-Z0-9]+]]: tensor<6x4xi32>
// CHECK-NOT:     iree_linalg_ext.general_scatter
// CHECK:         return %[[ORIGINAL0]], %[[ORIGINAL1]]
func.func @general_scatter_no_updates(%updates0: tensor<3x0xi32>, %updates1: tensor<3x0xi32>, %indices: tensor<3x1xi64>, %original0: tensor<6x4xi32>, %original1: tensor<6x4xi32>) -> (tensor<6x4xi32>, tensor<6x4xi32>) {
  %result:2 = iree_linalg_ext.general_scatter <index_vector_dim = 1, update_window_dims = [1], inserted_window_dims = [0], input_batching_dims = [], scatter_indices_batching_dims = [], scatter_dims_to_operand_dims = [0]>
    ins(%updates0, %updates1 : tensor<3x0xi32>, tensor<3x0xi32>)
    indices(%indices : tensor<3x1xi64>)
    outs(%original0, %original1 : tensor<6x4xi32>, tensor<6x4xi32>) {
  ^bb0(%current0: i32, %current1: i32, %update0: i32, %update1: i32):
    iree_linalg_ext.yield %update0, %update1 : i32, i32
  } -> tensor<6x4xi32>, tensor<6x4xi32>
  return %result#0, %result#1 : tensor<6x4xi32>, tensor<6x4xi32>
}

// -----

// An effect in the reducer keeps an unused general_scatter alive.
// CHECK-LABEL: func.func @unused_effectful_general_scatter(
//       CHECK:   iree_linalg_ext.general_scatter
//       CHECK:     util.status.check_ok
func.func @unused_effectful_general_scatter(%updates: tensor<3x4xi32>, %indices: tensor<3x1xi32>, %original: tensor<5x4xi32>, %status: i32) {
  %0 = iree_linalg_ext.general_scatter <update_window_dims = [1], inserted_window_dims = [0], input_batching_dims = [], scatter_indices_batching_dims = [], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>
      ins(%updates : tensor<3x4xi32>) indices(%indices : tensor<3x1xi32>) outs(%original : tensor<5x4xi32>) {
  ^bb0(%current: i32, %update: i32):
    util.status.check_ok %status, "reducer check"
    iree_linalg_ext.yield %update : i32
  } -> tensor<5x4xi32>
  return
}
