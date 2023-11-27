// RUN: iree-opt --iree-codegen-decompose-batch-mmt4d-ops --split-input-file %s | FileCheck %s

func.func @batch_mmt4d_with_fill(%arg0: tensor<1x10x32x8x1xf32>, %arg1: tensor<1x80x32x4x1xf32>, %arg2: tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x10x32x8x1xf32>, tensor<1x80x32x4x1xf32>) outs(%0 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  return %1 : tensor<1x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_fill
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[FILL]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<1x10x80x8x4xf32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xf32>

// -----

func.func @batch_mmt4d_with_no_fill(%arg0: tensor<1x10x32x8x1xf32>, %arg1: tensor<1x80x32x4x1xf32>, %arg2: tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32> {
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x10x32x8x1xf32>, tensor<1x80x32x4x1xf32>) outs(%arg2 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  return %1 : tensor<1x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_no_fill
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<1x10x80x8x4xf32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xf32>

// -----

func.func @batch_mmt4d_with_extened_inputs(%arg0: tensor<1x10x32x8x1xi8>, %arg1: tensor<1x80x32x4x1xi8>, %arg2: tensor<1x10x80x8x4xi32>) -> tensor<1x10x80x8x4xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x10x32x8x1xi32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, 
                                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], 
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} 
                       ins(%arg0 : tensor<1x10x32x8x1xi8>) outs(%0 : tensor<1x10x32x8x1xi32>) {
  ^bb0(%in: i8, %out: i32):
    %6 = arith.extsi %in : i8 to i32
    linalg.yield %6 : i32
  } -> tensor<1x10x32x8x1xi32>
  %2 = tensor.empty() : tensor<1x80x32x4x1xi32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], 
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
                       ins(%arg1 : tensor<1x80x32x4x1xi8>) outs(%2 : tensor<1x80x32x4x1xi32>) {
  ^bb0(%in: i8, %out: i32):
    %6 = arith.extsi %in : i8 to i32
    linalg.yield %6 : i32
  } -> tensor<1x80x32x4x1xi32>
  %4 = linalg.fill ins(%c0_i32 : i32) outs(%arg2 : tensor<1x10x80x8x4xi32>) -> tensor<1x10x80x8x4xi32>
  %5 = linalg.batch_mmt4d ins(%1, %3 : tensor<1x10x32x8x1xi32>, tensor<1x80x32x4x1xi32>) outs(%4 : tensor<1x10x80x8x4xi32>) -> tensor<1x10x80x8x4xi32>
  return %5 : tensor<1x10x80x8x4xi32>
}

// CHECK:      #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:      func.func @batch_mmt4d_with_extened_inputs
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xi8>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xi8>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xi32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0 : i32
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xi32> to tensor<10x80x8x4xi32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[CST]] : i32) outs(%[[EXT_OUT]] : tensor<10x80x8x4xi32>) -> tensor<10x80x8x4xi32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xi8> to tensor<10x32x8x1xi8>
// CHECK-DAG:    %[[INIT_GEN_LHS:.+]] = tensor.empty() : tensor<10x32x8x1xi32>
// CHECK-DAG:    %[[GEN_LHS:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXT_LHS]] : tensor<10x32x8x1xi8>) outs(%[[INIT_GEN_LHS]] : tensor<10x32x8x1xi32>) {
// CHECK-NEXT:       ^bb0(%[[ARGIN_GEN_LHS:.+]]: i8, %[[ARGOUT_GEN_LHS:.+]]: i32):
// CHECK-NEXT:         %[[EXT_GEN_LHS:.+]] = arith.extsi %[[ARGIN_GEN_LHS]] : i8 to i32
// CHECK-NEXT:         linalg.yield %[[EXT_GEN_LHS]] : i32
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xi8> to tensor<80x32x4x1xi8>
// CHECK-DAG:    %[[INIT_GEN_RHS:.+]] = tensor.empty() : tensor<80x32x4x1xi32>
// CHECK-DAG:    %[[GEN_RHS:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXT_RHS]] : tensor<80x32x4x1xi8>) outs(%[[INIT_GEN_RHS]] : tensor<80x32x4x1xi32>) {
// CHECK-NEXT:       ^bb0(%[[ARGIN_GEN_RHS:.+]]: i8, %[[ARGOUT_GEN_RHS:.+]]: i32):
// CHECK-NEXT:         %[[EXT_GEN_RHS:.+]] = arith.extsi %[[ARGIN_GEN_RHS]] : i8 to i32
// CHECK-NEXT:         linalg.yield %[[EXT_GEN_RHS]] : i32
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[GEN_LHS]], %[[GEN_RHS]] : tensor<10x32x8x1xi32>, tensor<80x32x4x1xi32>) outs(%[[FILL]] : tensor<10x80x8x4xi32>) -> tensor<10x80x8x4xi32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xi32> into tensor<1x10x80x8x4xi32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xi32>

// -----

func.func @batch_mmt4d_with_fill_batch_dim(%arg0: tensor<12x10x32x8x1xf32>, %arg1: tensor<12x80x32x4x1xf32>, %arg2: tensor<12x10x80x8x4xf32>) -> tensor<12x10x80x8x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<12x10x80x8x4xf32>) -> tensor<12x10x80x8x4xf32>
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<12x10x32x8x1xf32>, tensor<12x80x32x4x1xf32>) outs(%0 : tensor<12x10x80x8x4xf32>) -> tensor<12x10x80x8x4xf32>
  return %1 : tensor<12x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_fill_batch_dim
// CHECK-SAME:   %[[LHS:.+]]: tensor<12x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<12x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<12x10x80x8x4xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C12:.+]] = arith.constant 12 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[TILED_RES:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C12]] step %[[C1]] iter_args(%[[OUTPUT:.+]] = %[[OUT]]) -> (tensor<12x10x80x8x4xf32>) {
// CHECK-DAG:      %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUTPUT]][%[[IV]], 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<12x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK:          %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK-DAG:      %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][%[[IV]], 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<12x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:      %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][%[[IV]], 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<12x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:          %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[FILL]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:          %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUTPUT]][%[[IV]], 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<12x10x80x8x4xf32>
// CHECK:          scf.yield %[[INS]] : tensor<12x10x80x8x4xf32>
// CHECK:        }
// CHECK:        return %[[TILED_RES]] : tensor<12x10x80x8x4xf32>

