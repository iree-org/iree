// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

func.func @pack_canonicalize(%arg0 : tensor<?x?xi32>,
    %arg1 : tensor<1x2x3x3xi32>) -> tensor<1x?x3x3xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.cast %arg1 : tensor<1x2x3x3xi32> to tensor<1x?x3x3xi32>
  %1 = iree_linalg_ext.pack %arg0 padding_value(%c0_i32 : i32)
      inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %0
      : (tensor<?x?xi32> tensor<1x?x3x3xi32>) -> tensor<1x?x3x3xi32>
  return %1 : tensor<1x?x3x3xi32>
}
// CHECK-LABEL: func.func @pack_canonicalize
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xi32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<1x2x3x3xi32>
//       CHECK:   %[[PAD_VALUE:.+]] = arith.constant 0 : i32
//       CHECK:   %[[PACK:.+]] = iree_linalg_ext.pack %[[ARG0]]
//  CHECK-SAME:       padding_value(%[[PAD_VALUE]] : i32)
//  CHECK-SAME:       into %[[ARG1]]
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[PACK]]
//       CHECK:   return %[[CAST]]

// -----

func.func @sort_drop_unused_results(%arg0 : tensor<?x10xf32>,
    %arg1 : tensor<?x10xi64>) -> tensor<?x10xf32> {
  %0:2 = iree_linalg_ext.sort dimension(1) outs(%arg0, %arg1: tensor<?x10xf32>,
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
//       CHECK:   %[[SORT:.+]] = iree_linalg_ext.sort dimension(1) outs(%[[ARG0]] : tensor<?x10xf32>)

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
