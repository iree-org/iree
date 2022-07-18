// RUN: iree-opt --split-input-file --pass-pipeline="func.func(iree-codegen-decompose-linalg-generic)" %s | FileCheck %s

// CHECK-LABEL: @single_scalar_op_no_ops
func.func @single_scalar_op_no_ops(%arg0 : tensor<64xf32>, %arg1 : tensor<?x64xf32>) -> tensor<?x64xf32> {
  // CHECK: %[[G0:.*]] = linalg.generic
  // CHECK: %[[S0:.*]] = arith.addf
  // CHECK: linalg.yield %[[S0]]
  // CHECK: return %[[G0]]
  %12 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%arg1 : tensor<?x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %13 = arith.addf %arg2, %arg3 : f32
    linalg.yield %13 : f32
  } -> tensor<?x64xf32>
  func.return %12 : tensor<?x64xf32>
}

// -----
// CHECK-LABEL: @direct_yield_no_ops
func.func @direct_yield_no_ops(%arg0 : tensor<64xf32>, %arg1 : tensor<?x64xf32>) -> tensor<?x64xf32> {
  // CHECK: %[[G0:.*]] = linalg.generic
  // CHECK-NOT: linalg.generic
  // CHECK: linalg.yield
  // CHECK: return %[[G0]]
  %12 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%arg1 : tensor<?x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    linalg.yield %arg3 : f32
  } -> tensor<?x64xf32>
  func.return %12 : tensor<?x64xf32>
}

// -----
// CHECK: #map0 = affine_map<(d0, d1) -> (d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @parallel_with_broadcast_dynamic
func.func @parallel_with_broadcast_dynamic(
    %arg0 : tensor<64xf32>, %arg1 : tensor<?x64xf32>) -> tensor<?x64xf32> {
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: %[[D0:.*]] = tensor.dim %arg1, %[[C0]] : tensor<?x64xf32>
  //
  // Generic0: Decomposed addf
  //      CHECK: %[[IT0:.*]] = linalg.init_tensor [%[[D0]], 64] : tensor<?x64xf32>
  //      CHECK: %[[G0:.*]] = linalg.generic
  // CHECK-SAME:   {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:   ins(%arg0, %arg1 : tensor<64xf32>, tensor<?x64xf32>) outs(%[[IT0]] : tensor<?x64xf32>)
  //      CHECK:   ^bb0(%[[BARG0_0:.*]]: f32, %[[BARG_0_1:.*]]: f32, %[[BARG_0_2:.*]]: f32):
  //      CHECK:     %[[V0:.*]] = arith.addf %[[BARG0_0]], %[[BARG_0_1]] : f32
  //      CHECK:     linalg.yield %[[V0]]
  //
  // Generic1: Decomposed mulf
  //      CHECK: %[[IT1:.*]] = linalg.init_tensor [%[[D0]], 64] : tensor<?x64xf32>
  //      CHECK: %[[G1:.*]] = linalg.generic
  // CHECK-SAME:   {indexing_maps = [#map1, #map0, #map1], iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:   ins(%[[G0]], %arg0 : tensor<?x64xf32>, tensor<64xf32>) outs(%[[IT1]] : tensor<?x64xf32>)
  //      CHECK:   ^bb0(%[[BARG1_0:.*]]: f32, %[[BARG_1_1:.*]]: f32, %[[BARG_1_2:.*]]: f32):
  //      CHECK:     %[[V1:.*]] = arith.mulf %[[BARG1_0]], %[[BARG_1_1]] : f32
  //      CHECK:     linalg.yield %[[V1]]
  //
  // Degenerate final copy.
  // Note that canonicalization will collapse this with the previous
  // in this case but emitting keeps us fully general.
  //      CHECK: %[[G2:.*]] = linalg.generic
  // CHECK-SAME:   {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME:   ins(%[[G1]] : tensor<?x64xf32>) outs(%arg1 : tensor<?x64xf32>)
  //      CHECK:   ^bb0(%[[BARG2_0:.*]]: f32, %[[BARG2_1:.*]]: f32):
  //      CHECK:     linalg.yield %[[BARG2_0]]
  //      CHECK: return %[[G2]]
  %12 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%arg1 : tensor<?x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %13 = arith.addf %arg2, %arg3 : f32
    %14 = arith.mulf %13, %arg2 : f32
    linalg.yield %14 : f32
  } -> tensor<?x64xf32>
  func.return %12 : tensor<?x64xf32>
}
