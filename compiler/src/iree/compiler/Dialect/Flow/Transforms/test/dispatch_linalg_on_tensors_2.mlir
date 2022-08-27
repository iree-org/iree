// RUN: iree-opt --split-input-file --pass-pipeline="func.func(iree-flow-dispatch-linalg-on-tensors-pass)"

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @softmax(%arg0: tensor<12x128x128xf32>) -> tensor<12x128x128xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %0 = linalg.init_tensor [12, 128] : tensor<12x128xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<12x128xf32>) -> tensor<12x128xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<12x128x128xf32>) outs(%1 : tensor<12x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %7 : f32
    } -> tensor<12x128xf32>
    %3 = linalg.init_tensor [12, 128, 128] : tensor<12x128x128xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<12x128xf32>) -> tensor<12x128xf32>
    %5:2 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %2 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%3, %4 : tensor<12x128x128xf32>, tensor<12x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %7 = arith.subf %arg1, %arg2 : f32
      %8 = math.exp %7 : f32
      %9 = arith.addf %8, %arg4 : f32
      linalg.yield %8, %9 : f32, f32
    } -> (tensor<12x128x128xf32>, tensor<12x128xf32>)
    %6 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5#0, %5#1 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%3 : tensor<12x128x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %7 = arith.divf %cst, %arg2 : f32
      %8 = arith.mulf %arg1, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<12x128x128xf32>
    return %6 : tensor<12x128x128xf32>
  }
}
// CHECK-LABEL: func @softmax(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12x128x128xf32>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[ARG0]])
//  CHECK-NEXT:     %[[ARG1:.+]]: !flow.dispatch.tensor<readonly:12x128x128xf32>
//       CHECK:     %[[LOAD0:.+]] = flow.dispatch.tensor.load %[[ARG1]]
//       CHECK:     %[[FILL0:.+]] = linalg.fill
//       CHECK:     %[[FILL1:.+]] = linalg.fill
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD0]] :
//       CHECK:     %[[GENERIC1:.+]]:2 = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD0]], %[[GENERIC0]] :
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC1]]#0, %[[GENERIC1]]#1 :
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC2]]
//       CHECK:     flow.return
//       CHECK:   return %[[DISPATCH]]

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4, d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func @batchnorm_training(%arg0: tensor<12xf32>, %arg1: tensor<12x12x12x12x12xf32>, %arg2: tensor<12xf32>) -> (tensor<12xf32>, tensor<12xf32>, tensor<12xf32>) {
    %cst = arith.constant 1.420000e+00 : f32
    %cst_0 = arith.constant 1.450000e+00 : f32
    %cst_1 = arith.constant 1.300000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [12] : tensor<12xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<12xf32>) -> tensor<12xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x12x12x12x12xf32>, tensor<12xf32>) outs(%1 : tensor<12xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %4 = arith.subf %arg3, %arg4 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.addf %arg5, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<12xf32>
    %3:3 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %2 : tensor<12xf32>, tensor<12xf32>) outs(%0, %0, %0 : tensor<12xf32>, tensor<12xf32>, tensor<12xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32):
      %4 = arith.divf %arg4, %cst_0 : f32
      %5 = arith.addf %4, %cst_1 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.subf %arg3, %6 : f32
      %8 = arith.mulf %7, %cst : f32
      %9 = arith.subf %arg3, %8 : f32
      linalg.yield %5, %6, %9 : f32, f32, f32
    } -> (tensor<12xf32>, tensor<12xf32>, tensor<12xf32>)
    return %3#0, %3#1, %3#2 : tensor<12xf32>, tensor<12xf32>, tensor<12xf32>
  }
}
// CHECK-LABEL: func @batchnorm_training(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<12x12x12x12x12xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<12xf32>
//       CHECK:   %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[ARG1]], %[[ARG2]], %[[ARG0]])
//  CHECK-NEXT:     %[[ARG3:.+]]: !flow.dispatch.tensor<readonly:12x12x12x12x12xf32>
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:12xf32>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:12xf32>
//   CHECK-DAG:     %[[LOAD0:.+]] = flow.dispatch.tensor.load %[[ARG3]]
//   CHECK-DAG:     %[[LOAD1:.+]] = flow.dispatch.tensor.load %[[ARG4]]
//   CHECK-DAG:     %[[LOAD2:.+]] = flow.dispatch.tensor.load %[[ARG5]]
//       CHECK:     %[[FILL:.+]] = linalg.fill
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD0]], %[[LOAD1]] :
//       CHECK:     %[[GENERIC1:.+]]:3 = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD2]], %[[GENERIC0]] :
//   CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC1]]#0
//   CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC1]]#1
//   CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC1]]#2
//       CHECK:     flow.return
//       CHECK:   return %[[DISPATCH]]#0, %[[DISPATCH]]#1, %[[DISPATCH]]#2
