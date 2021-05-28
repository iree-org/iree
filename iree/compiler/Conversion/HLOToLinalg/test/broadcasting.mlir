// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors='direct-hlo-client-lowering' %s | IreeFileCheck %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics. Since the broadcasting
// implementation lowers through hlo ops, we are primarily checking broadcast
// semantics and not exhaustively checking that the non broadcasting ops lower
// to the right linalg sequences.

// CHECK-LABEL: @addWithoutBroadcast
func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: outs(%0 : tensor<4xf32>
  // CHECK: addf
  // CHECK-NOT: linalg.generic
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK: #map0 = affine_map<(d0, d1) -> (d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @dynamicBroadcast
func @dynamicBroadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Should broadcast %arg0 -> %arg1 and assert on dynamic expansion.

  // CHECK: %[[C0_0:.*]] = constant 0 : index
  // CHECK: %[[ARG0_D0:.*]] = memref.dim %arg0, %[[C0_0]]
  // CHECK: %[[C0_1:.*]] = constant 0 : index
  // CHECK: %[[ARG1_D0:.*]] = memref.dim %arg1, %[[C0_1]] : tensor<?x?xf32>
  // CHECK: %[[C1_0:.*]] = constant 1 : index
  // CHECK: %[[ARG1_D1:.*]] = memref.dim %arg1, %[[C1_0]] : tensor<?x?xf32>
  // CHECK: %[[EQ:.*]] = cmpi eq, %[[ARG0_D0]], %[[ARG1_D1]] : index
  // CHECK: assert %[[EQ]], "mismatched dynamic broadcast extents"

  // CHECK: %[[GE:.*]] = cmpi sge, %[[ARG0_D0]], %[[ARG1_D1]] : index
  // CHECK: %[[BCAST_D1:.*]] = select %[[GE]], %[[ARG0_D0]], %[[ARG1_D1]] : index
  // CHECK: %[[INIT_0:.*]] = linalg.init_tensor [%[[ARG1_D0]], %[[BCAST_D1]]] : tensor<?x?xf32>
  // CHECK: %[[BCAST_ARG0:.*]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
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
func @dynamicNonScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarByScalarBroadcastDimensions
func @dynamicNonScalarByScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<f32>) -> tensor<1x4xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcastComplex
func @dynamicBroadcastComplex(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  // NOTE: The lowering specifically allows mhlo.complex through and this should
  // reduce to that.
  // CHECK: mhlo.complex
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @dynamicBroadcastCompare
func @dynamicBroadcastCompare(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  // NOTE: compare is unique because of the element type switch. The pattern
  // will fail or the verifier will catch it if wrong.
  // CHECK-NOT: mhlo.compare
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----
// CHECK-LABEL: func @selectv2
func @selectv2(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // All same type: should just short-circtuit to one mhlo.select / one generic.
  // CHECK: linalg.generic
  // CHECK:   %[[BODY:.*]] = select
  // CHECK-NOT: linalg.generic
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----
// CHECK: #map0 = affine_map<(d0) -> ()>
// CHECK: #map1 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @selectv2_pred_scalar
func @selectv2_pred_scalar(%arg0: tensor<i1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: %[[INIT_0:.*]] = linalg.init_tensor [2] : tensor<2xi1>
  // CHECK: %[[BCAST_PRED:.*]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<i1>) outs(%[[INIT_0]] : tensor<2xi1>)
  // CHECK: %[[INIT_1:.*]] = linalg.init_tensor [2] : tensor<2xi32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[BCAST_PRED]], %arg1, %arg2 : tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) outs(%[[INIT_1]] : tensor<2xi32>)
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----
// Note that broadcast_add is used as a proxy for all of the template
// expansions. Tests below merely verify that the op has an expansion.
// CHECK-LABEL: @andWithoutBroadcast
func @andWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.and
  %0 = chlo.broadcast_and %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @atan2WithoutBroadcast
func @atan2WithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.atan2
  %0 = chlo.broadcast_atan2 %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @compareWithoutBroadcast
func @compareWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.compare
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @complexWithoutBroadcast
func @complexWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xcomplex<f32>> {
  // NOTE: The lowering specifically allows mhlo.complex through and this should
  // reduce to that.
  // CHECK: mhlo.complex
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// -----
// CHECK-LABEL: @divideWithoutBroadcast
func @divideWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.divide
  %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @maximumWithoutBroadcast
func @maximumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.maximum
  %0 = chlo.broadcast_maximum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @minimumWithoutBroadcast
func @minimumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.minimum
  %0 = chlo.broadcast_minimum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @multiplyWithoutBroadcast
func @multiplyWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.multiply
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @orWithoutBroadcast
func @orWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.or
  %0 = chlo.broadcast_or %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @powerWithoutBroadcast
func @powerWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.power
  %0 = chlo.broadcast_power %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @remainderWithoutBroadcast
func @remainderWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.remainder
  %0 = chlo.broadcast_remainder %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @subWithoutBroadcast
func @subWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: mhlo.subtract
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @xorWithoutBroadcast
func @xorWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK-NOT: mhlo.xor
  %0 = chlo.broadcast_xor %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @ZetaWithoutBroadcast
func @ZetaWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
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
func @PolygammaWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> tensor<4xf32> {
  // This is a composition: it should lower completely.
  // CHECK-NOT: mhlo.
  %0 = chlo.broadcast_polygamma %arg0, %arg1
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
