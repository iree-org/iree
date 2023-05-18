// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors %s | FileCheck %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics. Since the broadcasting
// implementation lowers through mhlo ops, we are primarily checking broadcast
// semantics and not exhaustively checking that the non broadcasting ops lower
// to the right linalg sequences.

// CHECK-LABEL: @addWithoutBroadcast
func.func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: outs(%0 : tensor<4xf32>
  // CHECK: addf
  // CHECK-NOT: linalg.generic
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK: #map = affine_map<(d0, d1) -> (d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @dynamicBroadcast
func.func @dynamicBroadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Should broadcast %arg0 -> %arg1 and cf.assert on dynamic expansion.

  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[ARG0_D0:.*]] = tensor.dim %arg0, %[[C0]]
  // CHECK-DAG: %[[ARG1_D0:.*]] = tensor.dim %arg1, %[[C0]] : tensor<?x?xf32>
  // CHECK-DAG: %[[ARG1_D1:.*]] = tensor.dim %arg1, %[[C1]] : tensor<?x?xf32>
  // CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[ARG0_D0]], %[[ARG1_D1]] : index
  // CHECK: cf.assert %[[EQ]], "mismatched dynamic broadcast extents"

  // CHECK: %[[INIT_0:.*]] = tensor.empty(%[[ARG1_D0]], %[[ARG0_D0]]) : tensor<?x?xf32>
  // CHECK: %[[BCAST_ARG0:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME: ins(%arg0 : tensor<?xf32>) outs(%[[INIT_0]] : tensor<?x?xf32>)

  // CHECK: %[[RESULT:.*]] = linalg.generic
  // CHECK-SAME: ins(%[[BCAST_ARG0]], %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)

  // CHECK-NOT: mhlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarBroadcastDimensions
func.func @dynamicNonScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarByScalarBroadcastDimensions
func.func @dynamicNonScalarByScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<f32>) -> tensor<1x4xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcastComplex
func.func @dynamicBroadcastComplex(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK-NOT: mhlo.complex
  // CHECK-NOT: chlo.broadcast_complex
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>

  %1 = "mhlo.real"(%0) : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32>
  %2 = "mhlo.imag"(%0) : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32>

  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcastCompare
func.func @dynamicBroadcastCompare(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  // NOTE: compare is unique because of the element type switch. The pattern
  // will fail or the verifier will catch it if wrong.
  // CHECK-NOT: mhlo.compare
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction EQ>} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----
// CHECK-LABEL: func.func @selectv2
func.func @selectv2(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // All same type: should just short-circtuit to one mhlo.select / one generic.
  // CHECK: linalg.generic
  // CHECK:   %[[BODY:.*]] = arith.select
  // CHECK-NOT: linalg.generic
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----
// CHECK: #map = affine_map<(d0) -> ()>
// CHECK: #map1 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @selectv2_pred_scalar
func.func @selectv2_pred_scalar(%arg0: tensor<i1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: %[[INIT_0:.*]] = tensor.empty() : tensor<2xi1>
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<i1>) outs(%[[INIT_0]] : tensor<2xi1>)
  // CHECK: %[[INIT_1:.*]] = tensor.empty() : tensor<2xi32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[BCAST_PRED]], %arg1, %arg2 : tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) outs(%[[INIT_1]] : tensor<2xi32>)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----
// CHECK: #map = affine_map<(d0, d1, d2) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d1, 0)>
// CHECK-LABEL: func.func @selectv2_broadcast_then
func.func @selectv2_broadcast_then(%arg0: tensor<i1>, %arg1: tensor<8x1xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<i1>)
  // CHECK: %[[BCAST_THEN:.*]] = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<8x1xi32>)
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[BCAST_PRED]], %[[BCAST_THEN]], %arg2 : tensor<2x8x8xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>)
  // CHECK: arith.select
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<8x1xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// -----
// CHECK: #map = affine_map<(d0, d1, d2) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d1, 0)>
// CHECK-LABEL: func.func @selectv2_broadcast_else
func.func @selectv2_broadcast_else(%arg0: tensor<i1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<8x1xi32>) -> tensor<2x8x8xi32> {
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<i1>)
  // CHECK: %[[BCAST_ELSE:.*]] = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8x1xi32>)
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[BCAST_PRED]], %arg1, %[[BCAST_ELSE]] : tensor<2x8x8xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>)
  // CHECK: arith.select
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x8x8xi32>, tensor<8x1xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// -----
// CHECK: #map = affine_map<(d0, d1, d2) -> (0)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @selectv2_broadcast_pred
func.func @selectv2_broadcast_pred(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1xi1>)
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[BCAST_PRED]], %arg1, %arg2 : tensor<2x8x8xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>)
  // CHECK: arith.select
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// -----
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, 0, 0)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
// CHECK: #map3 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
// CHECK-LABEL: func.func @selectv2_broadcast_all
func.func @selectv2_broadcast_all(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<8x8x8xi32> {
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x1x1xi1>)
  // CHECK: %[[BCAST_THEN:.*]] = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x8x1xi32>)
  // CHECK: %[[BCAST_ELSE:.*]] = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x1x8xi32>)
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[BCAST_PRED]], %[[BCAST_THEN]], %[[BCAST_ELSE]] : tensor<8x8x8xi1>, tensor<8x8x8xi32>, tensor<8x8x8xi32>)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  return %0: tensor<8x8x8xi32>
}

// -----
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, 0, 0)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
// CHECK: #map3 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
// CHECK-LABEL: func.func @selectv2_broadcast_dyn_pred
func.func @selectv2_broadcast_dyn_pred(%arg0: tensor<?x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<?x8x8xi32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM_PRED_0:.*]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INIT_PRED:.*]] = tensor.empty(%[[DIM_PRED_0]])
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map, #map1]
  //     CHECK-SAME: ins(%arg0 : tensor<?x1x1xi1>) outs(%[[INIT_PRED]] : tensor<?x8x8xi1>)
  // CHECK: %[[INIT_THEN:.*]] = tensor.empty(%[[DIM_PRED_0]])
  // CHECK: %[[BCAST_THEN:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map2, #map1]
  //     CHECK-SAME: ins(%arg1 : tensor<1x8x1xi32>) outs(%[[INIT_THEN]] : tensor<?x8x8xi32>)
  // CHECK: %[[INIT_ELSE:.*]] = tensor.empty(%[[DIM_PRED_0]])
  // CHECK: %[[BCAST_ELSE:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map3, #map1]
  //     CHECK-SAME: ins(%arg2 : tensor<1x1x8xi32>) outs(%[[INIT_ELSE]] : tensor<?x8x8xi32>)
  // CHECK: %[[SHAPE_BCAST_THEN:.*]] = shape.shape_of %[[BCAST_THEN]]
  // CHECK: %[[DIM_BCAST_THEN_0:.*]] = tensor.extract %[[SHAPE_BCAST_THEN]][%[[C0]]]
  // CHECK: %[[INIT_RESULT:.*]] = tensor.empty(%[[DIM_BCAST_THEN_0]])
  // CHECK: linalg.generic
  //     CHECK-SAME: ins(%[[BCAST_PRED]], %[[BCAST_THEN]], %[[BCAST_ELSE]] : tensor<?x8x8xi1>, tensor<?x8x8xi32>, tensor<?x8x8xi32>) outs(%[[INIT_RESULT]] : tensor<?x8x8xi32>)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<?x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x8xi32>) -> tensor<?x8x8xi32>
  return %0: tensor<?x8x8xi32>
}

// -----
// CHECK-LABEL: func.func @selectv2_broadcast_dyn_then
func.func @selectv2_broadcast_dyn_then(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x?x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<8x?x8xi32> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM_THEN_1:.*]] = tensor.dim %arg1, %[[C1]]
  // CHECK: %[[INIT_PRED:.*]] = tensor.empty(%[[DIM_THEN_1]])
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map, #map1]
  //     CHECK-SAME: ins(%arg0 : tensor<8x1x1xi1>) outs(%[[INIT_PRED]] : tensor<8x?x8xi1>)
  // CHECK: %[[INIT_THEN:.*]] = tensor.empty(%[[DIM_THEN_1]])
  // CHECK: %[[BCAST_THEN:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map2, #map1]
  //     CHECK-SAME: ins(%arg1 : tensor<1x?x1xi32>) outs(%[[INIT_THEN]] : tensor<8x?x8xi32>)
  // CHECK: %[[INIT_ELSE:.*]] = tensor.empty(%[[DIM_THEN_1]])
  // CHECK: %[[BCAST_ELSE:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map3, #map1]
  //     CHECK-SAME: ins(%arg2 : tensor<1x1x8xi32>) outs(%[[INIT_ELSE]] : tensor<8x?x8xi32>)
  // CHECK: %[[SHAPE_BCAST_THEN:.*]] = shape.shape_of %[[BCAST_THEN]]
  // CHECK: %[[DIM_BCAST_THEN_1:.*]] = tensor.extract %[[SHAPE_BCAST_THEN]][%[[C1]]]
  // CHECK: %[[INIT_RESULT:.*]] = tensor.empty(%[[DIM_BCAST_THEN_1]])
  // CHECK: linalg.generic
  //     CHECK-SAME: ins(%[[BCAST_PRED]], %[[BCAST_THEN]], %[[BCAST_ELSE]] : tensor<8x?x8xi1>, tensor<8x?x8xi32>, tensor<8x?x8xi32>) outs(%[[INIT_RESULT]] : tensor<8x?x8xi32>)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x?x1xi32>, tensor<1x1x8xi32>) -> tensor<8x?x8xi32>
  return %0: tensor<8x?x8xi32>
}

// -----
// CHECK-LABEL: func.func @selectv2_broadcast_dyn_else
func.func @selectv2_broadcast_dyn_else(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x?xi32>) -> tensor<8x8x?xi32> {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[DIM_ELSE_2:.*]] = tensor.dim %arg2, %[[C2]]
  // CHECK: %[[INIT_PRED:.*]] = tensor.empty(%[[DIM_ELSE_2]])
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map, #map1]
  //     CHECK-SAME: ins(%arg0 : tensor<8x1x1xi1>) outs(%[[INIT_PRED]] : tensor<8x8x?xi1>)

  // CHECK: %[[INIT_THEN:.*]] = tensor.empty(%[[DIM_ELSE_2]])
  // CHECK: %[[BCAST_THEN:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map2, #map1]
  //     CHECK-SAME: ins(%arg1 : tensor<1x8x1xi32>) outs(%[[INIT_THEN]] : tensor<8x8x?xi32>)
  // CHECK: %[[INIT_ELSE:.*]] = tensor.empty(%[[DIM_ELSE_2]])
  // CHECK: %[[BCAST_ELSE:.*]] = linalg.generic
  //     CHECK-SAME: indexing_maps = [#map3, #map1]
  //     CHECK-SAME: ins(%arg2 : tensor<1x1x?xi32>) outs(%[[INIT_ELSE]] : tensor<8x8x?xi32>)
  // CHECK: %[[SHAPE_BCAST_THEN:.*]] = shape.shape_of %[[BCAST_THEN]]
  // CHECK: %[[DIM_BCAST_THEN_1:.*]] = tensor.extract %[[SHAPE_BCAST_THEN]][%[[C2]]]
  // CHECK: %[[INIT_RESULT:.*]] = tensor.empty(%[[DIM_BCAST_THEN_1]])
  // CHECK: linalg.generic
  //     CHECK-SAME: ins(%[[BCAST_PRED]], %[[BCAST_THEN]], %[[BCAST_ELSE]] : tensor<8x8x?xi1>, tensor<8x8x?xi32>, tensor<8x8x?xi32>) outs(%[[INIT_RESULT]] : tensor<8x8x?xi32>)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x?xi32>) -> tensor<8x8x?xi32>
  return %0: tensor<8x8x?xi32>
}

// -----
// CHECK-LABEL: func.func @selectv2_broadcast_dyn_all
func.func @selectv2_broadcast_dyn_all(%arg0: tensor<?x1x1xi1>, %arg1: tensor<?x8x1xi32>, %arg2: tensor<?x1x?xi32>) -> tensor<?x8x?xi32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[PRED_D0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x1x1xi1>
  // CHECK-DAG: %[[THEN_D0:.*]] = tensor.dim %arg1, %[[C0]] : tensor<?x8x1xi32>
  // CHECK-DAG: %[[ELSE_D0:.*]] = tensor.dim %arg2, %[[C0]] : tensor<?x1x?xi32>
  // CHECK-DAG: %[[ELSE_D2:.*]] = tensor.dim %arg2, %[[C2]] : tensor<?x1x?xi32>
  // CHECK: %[[CMP_0:.*]] = arith.cmpi eq, %[[PRED_D0]], %[[THEN_D0]] : index
  // CHECK: cf.assert %[[CMP_0]], "mismatched dynamic broadcast extents"
  // CHECK: %[[CMP_1:.*]] = arith.cmpi eq, %[[PRED_D0]], %[[ELSE_D0]] : index
  // CHECK: cf.assert %[[CMP_1]], "mismatched dynamic broadcast extents"
  // Only two cf.asserts are needed. The rest are statically verified.
  // CHECK-NOT: cf.assert
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<?x1x1xi1>, tensor<?x8x1xi32>, tensor<?x1x?xi32>) -> tensor<?x8x?xi32>
  return %0: tensor<?x8x?xi32>
}

// -----
// Note that broadcast_add is used as a proxy for all of the template
// expansions. Tests below merely verify that the op has an expansion.
// CHECK-LABEL: @andWithoutBroadcast
func.func @andWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.and
  %0 = chlo.broadcast_and %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @atan2WithoutBroadcast
func.func @atan2WithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.atan2
  %0 = chlo.broadcast_atan2 %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @compareWithoutBroadcast
func.func @compareWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.compare
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction EQ>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @complexWithoutBroadcast
func.func @complexWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-NOT: mhlo.complex
  // CHECK-NOT: chlo.broadcast_complex
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>

  %1 = "mhlo.real"(%0) : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %2 = "mhlo.imag"(%0) : (tensor<4xcomplex<f32>>) -> tensor<4xf32>

  return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

// -----
// CHECK-LABEL: @divideWithoutBroadcast
func.func @divideWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.divide
  %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @maximumWithoutBroadcast
func.func @maximumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.maximum
  %0 = chlo.broadcast_maximum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @minimumWithoutBroadcast
func.func @minimumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.minimum
  %0 = chlo.broadcast_minimum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @multiplyWithoutBroadcast
func.func @multiplyWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.multiply
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @orWithoutBroadcast
func.func @orWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.or
  %0 = chlo.broadcast_or %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @powerWithoutBroadcast
func.func @powerWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.power
  %0 = chlo.broadcast_power %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @remainderWithoutBroadcast
func.func @remainderWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.remainder
  %0 = chlo.broadcast_remainder %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @subWithoutBroadcast
func.func @subWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.subtract
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @xorWithoutBroadcast
func.func @xorWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.xor
  %0 = chlo.broadcast_xor %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @ZetaWithoutBroadcast
func.func @ZetaWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> tensor<4xf32> {
  // This is a composition: it should lower completely.
  // CHECK-NOT: mhlo.
  %0 = chlo.broadcast_zeta %arg0, %arg1
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @PolygammaWithoutBroadcast
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>)
func.func @PolygammaWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> tensor<4xf32> {
  // This is a composition: it should lower completely.
  // CHECK-NOT: mhlo.
  %0 = chlo.broadcast_polygamma %arg0, %arg1
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @fallbackDynamicReshape
func.func @fallbackDynamicReshape(%arg0 : tensor<4x?x3x?xui32>, %arg1 : tensor<5xindex>) -> tensor<12x?x?x1x?xui32> {
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[RESULT_D1:.*]] = tensor.extract %arg1[%[[C1]]] : tensor<5xindex>
  // CHECK-DAG: %[[RESULT_D2:.*]] = tensor.extract %arg1[%[[C2]]] : tensor<5xindex>
  // CHECK-DAG: %[[RESULT_D4:.*]] = tensor.extract %arg1[%[[C4]]] : tensor<5xindex>
  // CHECK-DAG: %[[ARG_D1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<4x?x3x?xi32>
  // CHECK-DAG: %[[ARG_D3:.*]] = tensor.dim %arg0, %[[C3]] : tensor<4x?x3x?xi32>
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<4x?x3x?xi32>{%[[ARG_D1]], %[[ARG_D3]]} -> tensor<12x?x?x1x?xi32>{%[[RESULT_D1]], %[[RESULT_D2]], %[[RESULT_D4]]}
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<4x?x3x?xui32>, tensor<5xindex>) -> tensor<12x?x?x1x?xui32>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<12x?x?x1x?xui32>
}

// -----
// CHECK-LABEL: @fallbackDynamicReshapeInt
func.func @fallbackDynamicReshapeInt(%arg0 : tensor<4x?x3x?xui32>, %arg1 : tensor<5xi32>) -> tensor<12x?x?x1x?xui32> {
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[D1:.*]] = tensor.extract %arg1[%[[C1]]] : tensor<5xi32>
  // CHECK-DAG: %[[D2:.*]] = tensor.extract %arg1[%[[C2]]] : tensor<5xi32>
  // CHECK-DAG: %[[D4:.*]] = tensor.extract %arg1[%[[C4]]] : tensor<5xi32>
  // CHECK-DAG: %[[RESULT_D1:.*]] = arith.index_cast %[[D1]] : i32 to index
  // CHECK-DAG: %[[RESULT_D2:.*]] = arith.index_cast %[[D2]] : i32 to index
  // CHECK-DAG: %[[RESULT_D4:.*]] = arith.index_cast %[[D4]] : i32 to index
  // CHECK-DAG: %[[ARG_D1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<4x?x3x?xi32>
  // CHECK-DAG: %[[ARG_D3:.*]] = tensor.dim %arg0, %[[C3]] : tensor<4x?x3x?xi32>
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<4x?x3x?xi32>{%[[ARG_D1]], %[[ARG_D3]]} -> tensor<12x?x?x1x?xi32>{%[[RESULT_D1]], %[[RESULT_D2]], %[[RESULT_D4]]}
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<4x?x3x?xui32>, tensor<5xi32>) -> tensor<12x?x?x1x?xui32>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<12x?x?x1x?xui32>
}
