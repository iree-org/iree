// RUN: iree-opt %s -pass-pipeline='builtin.module(func.func(iree-vector-ext-lower-transfer-gather-to-gather))' --split-input-file --mlir-print-local-scope | FileCheck %s

#map = affine_map<(d0)[s0] -> (0, 0, s0)>
#map1 = affine_map<(d0)[s0] -> (d0)>
module {
  func.func @lower_transfer_gather_to_vector_gather(%arg0: tensor<1x1x31xf32>, %arg1: tensor<1x1x1x1x16xf32>) -> tensor<1x1x1x1x16xf32> {
    %0 = ub.poison : vector<1x16xf32>
    %1 = ub.poison : vector<1x1x16xf32>
    %2 = ub.poison : vector<1x1x1x16xf32>
    %3 = ub.poison : vector<1x1x1x1x16xf32>
    %cst = arith.constant dense<2> : vector<16xindex>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %4 = vector.step : vector<16xindex>
    %5 = arith.muli %4, %cst : vector<16xindex>
    %6 = iree_vector_ext.transfer_gather %arg0[%c0, %c0, %c0] [%5 : vector<16xindex>], %cst_0 {indexing_maps = [#map, #map1]} : tensor<1x1x31xf32>, vector<16xf32>
    %7 = vector.insert_strided_slice %6, %0 {offsets = [0, 0], strides = [1]} : vector<16xf32> into vector<1x16xf32>
    %8 = vector.insert_strided_slice %7, %1 {offsets = [0, 0, 0], strides = [1, 1]} : vector<1x16xf32> into vector<1x1x16xf32>
    %9 = vector.insert_strided_slice %8, %2 {offsets = [0, 0, 0, 0], strides = [1, 1, 1]} : vector<1x1x16xf32> into vector<1x1x1x16xf32>
    %10 = vector.insert_strided_slice %9, %3 {offsets = [0, 0, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x1x16xf32> into vector<1x1x1x1x16xf32>
    %11 = vector.transfer_write %10, %arg1[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x16xf32>, tensor<1x1x1x1x16xf32>
    return %11 : tensor<1x1x1x1x16xf32>
  }
}

// CHECK-LABEL: func.func @lower_transfer_gather_to_vector_gather
// CHECK-SAME:  %[[SRC:.+]]: tensor<1x1x31xf32>
// CHECK-DAG:     %[[PASS_THRU:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:     %[[MASK:.+]] = arith.constant dense<true> : vector<16xi1>
// CHECK-DAG:     %[[STRIDE:.+]] = arith.constant dense<2> : vector<16xindex>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[STEP:.+]] = vector.step : vector<16xindex>
// CHECK:         %[[INDICES:.+]] = arith.muli %[[STEP]], %[[STRIDE]] : vector<16xindex>
// CHECK:         %[[GATHER:.+]] = vector.gather %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] [%[[INDICES]]], %[[MASK]], %[[PASS_THRU]]
// CHECK-SAME:      : tensor<1x1x31xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>
