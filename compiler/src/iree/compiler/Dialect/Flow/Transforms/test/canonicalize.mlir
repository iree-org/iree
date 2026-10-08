// Note: this file is for patterns explicitly added only during the
// flow-specific canonicalization pass. Canonicalization patterns registered on
// flow dialect ops should be tested under the appropriate
// iree/compiler/Dialect/Flow/IR/test/*_folding.mlir file for the op category.

// RUN: iree-opt --iree-flow-canonicalize %s --split-input-file --mlir-print-local-scope | FileCheck %s

util.func public @fold_full_insert_into_extract(
    %source: tensor<8x?xf32>,
    %dest: tensor<10x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %extract = tensor.extract_slice %dest [1, 1] [8, %size] [1, 1] : tensor<10x?xf32> to tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %extract [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @fold_full_insert_into_extract
//  CHECK-SAME:   %[[SOURCE:.+]]: tensor<8x?xf32>
//       CHECK:   util.return %[[SOURCE]]

// -----

util.func public @fold_full_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size) : tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @fold_full_insert_into_empty
//  CHECK-SAME:   %[[SOURCE:.+]]: tensor<8x?xf32>
//       CHECK:   util.return %[[SOURCE]]

// -----

util.func public @dont_fold_not_full_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size1: index, %size2: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size1) : tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size2] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @dont_fold_not_full_insert_into_empty
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   util.return %[[INSERT]]

// -----

util.func public @dont_fold_not_full_static_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size: index) -> tensor<10x?xf32> {
  %empty = tensor.empty(%size) : tensor<10x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<10x?xf32>
  util.return %insert : tensor<10x?xf32>
}

// CHECK-LABEL: util.func public @dont_fold_not_full_static_insert_into_empty
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   util.return %[[INSERT]]

// -----

util.func public @expand_affine(%arg0: index) -> index {
  %mul = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%arg0]
  util.return %mul : index
}

// CHECK-LABEL: util.func public @expand_affine
//  CHECK-SAME:   %[[ARG0:.+]]: index
//       CHECK:   %[[MUL:.+]] = arith.muli %[[ARG0]], %c4 overflow<nsw>
//       CHECK:   util.return %[[MUL]]

// -----

util.func public @fold_broadcast_with_empty_tensor() -> tensor<16x64xf32> {
  %input_empty = tensor.empty() : tensor<16xf32>
  %init = tensor.empty() : tensor<16x64xf32>
  %bcast = linalg.broadcast ins(%input_empty : tensor<16xf32>) outs(%init : tensor<16x64xf32>) dimensions = [1]
  util.return %bcast : tensor<16x64xf32>
}

// CHECK-LABEL: util.func public @fold_broadcast_with_empty_tensor
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<16x64xf32>
//       CHECK:   util.return %[[EMPTY]]

// -----

util.func public @fold_broadcast_generic_with_empty_tensor() -> tensor<16x64xf32> {
  %input_empty = tensor.empty() : tensor<16xf32>
  %init = tensor.empty() : tensor<16x64xf32>
  %bcast = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input_empty : tensor<16xf32>) outs(%init : tensor<16x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<16x64xf32>
  util.return %bcast : tensor<16x64xf32>
}

// CHECK-LABEL: util.func public @fold_broadcast_generic_with_empty_tensor
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<16x64xf32>
//       CHECK:   util.return %[[EMPTY]]
