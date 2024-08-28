// RUN: iree-opt --split-input-file --iree-dispatch-creation-bubble-up-extract-slices --iree-flow-canonicalize %s | FileCheck %s

util.func public @bubble_up_extract_rank_reduce(%arg0 : tensor<1024x7x7x2xi8>) -> tensor<1024x7x7xf32>{
  %0 = tensor.empty() : tensor<1024x7x7x2xf32>
  %cst = arith.constant 5.000000e-01 : f32
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1024x7x7x2xi8>) outs(%0 : tensor<1024x7x7x2xf32>) {
  ^bb0(%in: i8, %out: f32):
    %4 = arith.extsi %in : i8 to i32
    %5 = arith.sitofp %4 : i32 to f32
    %6 = arith.mulf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<1024x7x7x2xf32>

  %extracted_slice = tensor.extract_slice %1[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xf32> to tensor<1024x7x7xf32>
  util.return %extracted_slice : tensor<1024x7x7xf32>
}

// CHECK-LABEL:  @bubble_up_extract_rank_reduce
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:    %[[GENERIC:.+]] = linalg.generic
//       CHECK:    util.return %[[GENERIC]]

// -----

util.func public @bubble_up_extract(%arg0 : tensor<1024x7x7x2xi8>) -> tensor<1024x7x7x1xf32>{
  %0 = tensor.empty() : tensor<1024x7x7x2xf32>
  %cst = arith.constant 5.000000e-01 : f32
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1024x7x7x2xi8>) outs(%0 : tensor<1024x7x7x2xf32>) {
  ^bb0(%in: i8, %out: f32):
    %4 = arith.extsi %in : i8 to i32
    %5 = arith.sitofp %4 : i32 to f32
    %6 = arith.mulf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<1024x7x7x2xf32>

  %extracted_slice = tensor.extract_slice %1[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xf32> to tensor<1024x7x7x1xf32>
  util.return %extracted_slice : tensor<1024x7x7x1xf32>
}

// CHECK-LABEL:  @bubble_up_extract
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:    %[[GENERIC:.+]] = linalg.generic
//       CHECK:    util.return %[[GENERIC]]

// -----

util.func public @bubble_up_extract_multi_input(%arg0 : tensor<1024x7x7x2xi8>, %arg1 : tensor<1024x7x7x2xi8>) -> tensor<1024x7x7x1xf32>{
  %0 = tensor.empty() : tensor<1024x7x7x2xf32>
  %cst = arith.constant 5.000000e-01 : f32
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1024x7x7x2xi8>, tensor<1024x7x7x2xi8>) outs(%0 : tensor<1024x7x7x2xf32>) {
  ^bb0(%in: i8, %in_0 : i8, %out: f32):
    %4 = arith.extsi %in : i8 to i32
    %5 = arith.sitofp %4 : i32 to f32
    %6 = arith.mulf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<1024x7x7x2xf32>

  %extracted_slice = tensor.extract_slice %1[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xf32> to tensor<1024x7x7x1xf32>
  util.return %extracted_slice : tensor<1024x7x7x1xf32>
}

// CHECK-LABEL:  @bubble_up_extract_multi_input
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:    %[[EXTRACT0:.+]] = tensor.extract_slice %[[ARG0]]
//   CHECK-DAG:    %[[EXTRACT1:.+]] = tensor.extract_slice %[[ARG1]]
//       CHECK:    %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:      ins(%[[EXTRACT0]], %[[EXTRACT1]] : tensor<1024x7x7x1xi8>, tensor<1024x7x7x1xi8>)
//       CHECK:    util.return %[[GENERIC]]

// -----

util.func public @bubble_up_extract_with_use(%arg0 : tensor<1024x7x7x2xi8>) -> (tensor<1024x7x7xf32>, tensor<1024x7x7x2xf32>) {
  %0 = tensor.empty() : tensor<1024x7x7x2xf32>
  %cst = arith.constant 5.000000e-01 : f32
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1024x7x7x2xi8>) outs(%0 : tensor<1024x7x7x2xf32>) {
  ^bb0(%in: i8, %out: f32):
    %4 = arith.extsi %in : i8 to i32
    %5 = arith.sitofp %4 : i32 to f32
    %6 = arith.mulf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<1024x7x7x2xf32>

  %extracted_slice = tensor.extract_slice %1[0, 0, 0, 1] [1024, 7, 7, 1] [1, 1, 1, 1] : tensor<1024x7x7x2xf32> to tensor<1024x7x7xf32>
  util.return %extracted_slice, %1 : tensor<1024x7x7xf32>, tensor<1024x7x7x2xf32>
}

// CHECK-LABEL:  @bubble_up_extract_with_use
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//   CHECK-DAG:    %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:      ins(%[[ARG0]] : tensor<1024x7x7x2xi8>)
//
//   CHECK-DAG:    %[[EXTRACT0:.+]] = tensor.extract_slice %[[ARG0]]
//   CHECK-DAG:    %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:      ins(%[[EXTRACT0]] : tensor<1024x7x7xi8>)
//       CHECK:    util.return %[[GENERIC1]], %[[GENERIC0]]
