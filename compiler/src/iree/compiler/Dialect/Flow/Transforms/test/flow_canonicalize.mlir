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
