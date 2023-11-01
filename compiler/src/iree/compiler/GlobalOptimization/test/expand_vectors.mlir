// RUN: iree-opt --iree-global-opt-expand-vectors --split-input-file %s | FileCheck %s

func.func @vecmat_f32f32f32(%arg0 : tensor<250xf32>, %arg1 : tensor<250x100xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.vecmat ins(%arg0, %arg1 : tensor<250xf32>, tensor<250x100xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  return %0 : tensor<100xf32>
}
//      CHECK:  func @vecmat_f32f32f32(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<250xf32>, %[[ARG1:.+]]: tensor<250x100xf32>, %[[ARG2:.+]]: tensor<100xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<250xf32> into tensor<1x250xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<100xf32> into tensor<1x100xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[EXPANDED_IN]], %[[ARG1]] : tensor<1x250xf32>, tensor<250x100xf32>) outs(%[[EXPANDED_OUT]] : tensor<1x100xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<1x100xf32> into tensor<100xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @vecmat_bf16bf16f32_dynamic(%arg0 : tensor<?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.vecmat ins(%arg0, %arg1 : tensor<?xbf16>, tensor<?x?xbf16>)
      outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//      CHECK:  func @vecmat_bf16bf16f32_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?xbf16>, %[[ARG1:.+]]: tensor<?x?xbf16>, %[[ARG2:.+]]: tensor<?xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<?xbf16> into tensor<1x?xbf16>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[EXPANDED_IN]], %[[ARG1]] : tensor<1x?xbf16>, tensor<?x?xbf16>) outs(%[[EXPANDED_OUT]] : tensor<1x?xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @matvec_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  return %0 : tensor<100xf32>
}
//      CHECK:  func @matvec_f32f32f32(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<100x250xf32>, %[[ARG1:.+]]: tensor<250xf32>, %[[ARG2:.+]]: tensor<100xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1]] : tensor<250xf32> into tensor<250x1xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<100xf32> into tensor<100x1xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<100x250xf32>, tensor<250x1xf32>) outs(%[[EXPANDED_OUT]] : tensor<100x1xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<100x1xf32> into tensor<100xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @matvec_i8i8i32_dynamic(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?xi8>,
    %arg2 : tensor<?xi32>) -> tensor<?xi32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?xi8>)
      outs(%arg2 : tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
//      CHECK:  func @matvec_i8i8i32_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?xi8>, %[[ARG1:.+]]: tensor<?xi8>, %[[ARG2:.+]]: tensor<?xi32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1]] : tensor<?xi8> into tensor<?x1xi8>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<?xi32> into tensor<?x1xi32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<?x?xi8>, tensor<?x1xi8>) outs(%[[EXPANDED_OUT]] : tensor<?x1xi32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<?x1xi32> into tensor<?xi32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @batch_matvec_f32f32f32(%arg0 : tensor<3x100x250xf32>, %arg1 : tensor<3x250xf32>,
    %arg2 : tensor<3x100xf32>) -> tensor<3x100xf32> {
  %0 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<3x100x250xf32>, tensor<3x250xf32>)
      outs(%arg2 : tensor<3x100xf32>) -> tensor<3x100xf32>
  return %0 : tensor<3x100xf32>
}
//      CHECK:  func @batch_matvec_f32f32f32(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<3x100x250xf32>, %[[ARG1:.+]]: tensor<3x250xf32>, %[[ARG2:.+]]: tensor<3x100xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0], [1, 2]] : tensor<3x250xf32> into tensor<3x250x1xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0], [1, 2]] : tensor<3x100xf32> into tensor<3x100x1xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<3x100x250xf32>, tensor<3x250x1xf32>) outs(%[[EXPANDED_OUT]] : tensor<3x100x1xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0], [1, 2]] : tensor<3x100x1xf32> into tensor<3x100xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @batch_matvec_f16f16f16_dynamic(%arg0 : tensor<?x?x?xf16>, %arg1 : tensor<?x?xf16>,
    %arg2 : tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<?x?x?xf16>, tensor<?x?xf16>)
      outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}
//      CHECK:  func @batch_matvec_f16f16f16_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?x?xf16>, %[[ARG1:.+]]: tensor<?x?xf16>, %[[ARG2:.+]]: tensor<?x?xf16>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0], [1, 2]] : tensor<?x?xf16> into tensor<?x?x1xf16>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0], [1, 2]] : tensor<?x?xf16> into tensor<?x?x1xf16>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[ARG0]], %[[EXPANDED_IN]] : tensor<?x?x?xf16>, tensor<?x?x1xf16>) outs(%[[EXPANDED_OUT]] : tensor<?x?x1xf16>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0], [1, 2]] : tensor<?x?x1xf16> into tensor<?x?xf16>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @batch_vecmat_f32f32f32(%arg0 : tensor<3x250xf32>, %arg1 : tensor<3x250x100xf32>,
    %arg2 : tensor<3x100xf32>) -> tensor<3x100xf32> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<3x250xf32>, tensor<3x250x100xf32>)
      outs(%arg2 : tensor<3x100xf32>) -> tensor<3x100xf32>
  return %0 : tensor<3x100xf32>
}
//      CHECK:  func @batch_vecmat_f32f32f32(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<3x250xf32>, %[[ARG1:.+]]: tensor<3x250x100xf32>, %[[ARG2:.+]]: tensor<3x100xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2]] : tensor<3x250xf32> into tensor<3x1x250xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2]] : tensor<3x100xf32> into tensor<3x1x100xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[EXPANDED_IN]], %[[ARG1]] : tensor<3x1x250xf32>, tensor<3x250x100xf32>) outs(%[[EXPANDED_OUT]] : tensor<3x1x100xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1], [2]] : tensor<3x1x100xf32> into tensor<3x100xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @batch_vecmat_f16f32f32_dynamic(%arg0 : tensor<3x?xf16>, %arg1 : tensor<3x?x?xf32>,
    %arg2 : tensor<3x?xf32>) -> tensor<3x?xf32> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<3x?xf16>, tensor<3x?x?xf32>)
      outs(%arg2 : tensor<3x?xf32>) -> tensor<3x?xf32>
  return %0 : tensor<3x?xf32>
}
//      CHECK:  func @batch_vecmat_f16f32f32_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<3x?xf16>, %[[ARG1:.+]]: tensor<3x?x?xf32>, %[[ARG2:.+]]: tensor<3x?xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2]] : tensor<3x?xf16> into tensor<3x1x?xf16>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2]] : tensor<3x?xf32> into tensor<3x1x?xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[EXPANDED_IN]], %[[ARG1]] : tensor<3x1x?xf16>, tensor<3x?x?xf32>) outs(%[[EXPANDED_OUT]] : tensor<3x1x?xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1], [2]] : tensor<3x1x?xf32> into tensor<3x?xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @vecmat_bf16bf16f32_casted_dynamic(%arg0 : tensor<?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xbf16>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %casted0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, 
                                              affine_map<(d0) -> (d0)>], 
                             iterator_types = ["parallel"]} 
                             ins(%arg0 : tensor<?xbf16>) 
                             outs(%0 : tensor<?xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %2 = arith.extf %in : bf16 to f32
    linalg.yield %2 : f32
  } -> tensor<?xf32>
  %casted1 = arith.extf %arg1 : tensor<?x?xbf16> to tensor<?x?xf32>
  %1 = linalg.vecmat ins(%casted0, %casted1 : tensor<?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
//      CHECK:  func @vecmat_bf16bf16f32_casted_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?xbf16>, %[[ARG1:.+]]: tensor<?x?xbf16>, %[[ARG2:.+]]: tensor<?xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<?xbf16> into tensor<1x?xbf16>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
//  CHECK-DAG:  %[[CASTED0:.+]] = arith.extf %[[EXPANDED_IN]] : tensor<1x?xbf16> to tensor<1x?xf32>
//  CHECK-DAG:  %[[CASTED1:.+]] = arith.extf %[[ARG1]] : tensor<?x?xbf16> to tensor<?x?xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[CASTED0]], %[[CASTED1]] : tensor<1x?xf32>, tensor<?x?xf32>) outs(%[[EXPANDED_OUT]] : tensor<1x?xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
//      CHECK:  return %[[COLLAPSED]]

// -----

func.func @matvec_i8i8i32_casted_dynamic(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?xi8>,
    %arg2 : tensor<?xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?xi8>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %casted0 = arith.extui %arg0 : tensor<?x?xi8> to tensor<?x?xi32>
  %casted1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, 
                                              affine_map<(d0) -> (d0)>], 
                             iterator_types = ["parallel"]} 
                             ins(%arg1 : tensor<?xi8>) 
                             outs(%0 : tensor<?xi32>) {
  ^bb0(%in: i8, %out: i32):
    %2 = arith.extsi %in : i8 to i32
    linalg.yield %2 : i32
  } -> tensor<?xi32>
  %1 = linalg.matvec ins(%casted0, %casted1 : tensor<?x?xi32>, tensor<?xi32>)
      outs(%arg2 : tensor<?xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}
//      CHECK:  func @matvec_i8i8i32_casted_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?xi8>, %[[ARG1:.+]]: tensor<?xi8>, %[[ARG2:.+]]: tensor<?xi32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1]] : tensor<?xi8> into tensor<?x1xi8>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1]] : tensor<?xi32> into tensor<?x1xi32>
//  CHECK-DAG:  %[[CASTED0:.+]] = arith.extui %[[ARG0]] : tensor<?x?xi8> to tensor<?x?xi32>
//  CHECK-DAG:  %[[CASTED1:.+]] = arith.extsi %[[EXPANDED_IN]] : tensor<?x1xi8> to tensor<?x1xi32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.matmul ins(%[[CASTED0]], %[[CASTED1]] : tensor<?x?xi32>, tensor<?x1xi32>) outs(%[[EXPANDED_OUT]] : tensor<?x1xi32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1]] : tensor<?x1xi32> into tensor<?xi32>
//      CHECK:  return %[[COLLAPSED]]

func.func @batch_vecmat_casted_f16f32f32_dynamic(%arg0 : tensor<3x?xf16>, %arg1 : tensor<3x?x?xf32>,
    %arg2 : tensor<3x?xf32>) -> tensor<3x?xf32> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<3x?xf16>
  %0 = tensor.empty(%dim) : tensor<3x?xf32>
  %casted0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                                              affine_map<(d0, d1) -> (d0, d1)>], 
                             iterator_types = ["parallel", "parallel"]} 
                             ins(%arg0 : tensor<3x?xf16>) 
                             outs(%0 : tensor<3x?xf32>) {
  ^bb0(%in: f16, %out: f32):
    %2 = arith.extf %in : f16 to f32
    linalg.yield %2 : f32
  } -> tensor<3x?xf32>
  %1 = linalg.batch_vecmat ins(%casted0, %arg1 : tensor<3x?xf32>, tensor<3x?x?xf32>)
      outs(%arg2 : tensor<3x?xf32>) -> tensor<3x?xf32>
  return %1 : tensor<3x?xf32>
}

//      CHECK:  func @batch_vecmat_casted_f16f32f32_dynamic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<3x?xf16>, %[[ARG1:.+]]: tensor<3x?x?xf32>, %[[ARG2:.+]]: tensor<3x?xf32>
//  CHECK-DAG:  %[[EXPANDED_IN:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2]] : tensor<3x?xf16> into tensor<3x1x?xf16>
//  CHECK-DAG:  %[[CASTED0:.+]] = arith.extf %[[EXPANDED_IN]] : tensor<3x1x?xf16> to tensor<3x1x?xf32>
//  CHECK-DAG:  %[[EXPANDED_OUT:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2]] : tensor<3x?xf32> into tensor<3x1x?xf32>
//  CHECK-DAG:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[CASTED0]], %[[ARG1]] : tensor<3x1x?xf32>, tensor<3x?x?xf32>) outs(%[[EXPANDED_OUT]] : tensor<3x1x?xf32>)
//  CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MATMUL]] {{\[}}[0, 1], [2]] : tensor<3x1x?xf32> into tensor<3x?xf32>
//      CHECK:  return %[[COLLAPSED]]
