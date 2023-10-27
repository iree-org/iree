// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-generic-vectorization,iree-spirv-initial-vector-lowering,iree-codegen-hoist-redundant-vector-transfers,iree-spirv-final-vector-lowering))' \
// RUN:   %s | FileCheck %s

func.func @add(%lhs: tensor<2x8xf32>, %rhs: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %init = tensor.empty() : tensor<2x8xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<2x8xf32>, tensor<2x8xf32>)
    outs(%init : tensor<2x8xf32>) {
        ^bb0(%a : f32, %b : f32, %c : f32):
        %add = arith.addf %a, %b : f32
        %mul = arith.mulf %a, %b : f32
        %sub = arith.subf %add, %mul: f32
        linalg.yield %sub : f32
  } -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

//   CHECK-LABEL: func.func @add
// CHECK-COUNT-8:   vector.transfer_read %{{.+}} : tensor<2x8xf32>, vector<4xf32>
// CHECK-COUNT-4:   arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-4:   arith.mulf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-4:   arith.subf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-4:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x8xf32>

// -----

func.func @transpose_leading_one_dim(%input: tensor<4x1x1xf32>) -> tensor<1x1x4xf32> {
  %init = tensor.empty() : tensor<1x1x4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<4x1x1xf32>) outs(%init : tensor<1x1x4xf32>) {
  ^bb0(%arg0 : f32, %arg1 : f32):
    linalg.yield %arg0 : f32
  } -> tensor<1x1x4xf32>
  return %0: tensor<1x1x4xf32>
}

// CHECK-LABEL: func @transpose_leading_one_dim
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<4x1x1xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>

//       CHECK:   %[[R0:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]]{{.+}} : tensor<4x1x1xf32>, vector<1xf32>
//       CHECK:   %[[R1:.+]] = vector.transfer_read %[[INPUT]][%[[C1]], %[[C0]], %[[C0]]]{{.+}} : tensor<4x1x1xf32>, vector<1xf32>
//       CHECK:   %[[R2:.+]] = vector.transfer_read %[[INPUT]][%[[C2]], %[[C0]], %[[C0]]]{{.+}} : tensor<4x1x1xf32>, vector<1xf32>
//       CHECK:   %[[R3:.+]] = vector.transfer_read %[[INPUT]][%[[C3]], %[[C0]], %[[C0]]]{{.+}} : tensor<4x1x1xf32>, vector<1xf32>

//       CHECK:   %[[E0:.+]] = vector.extract %[[R0]][0] : f32 from vector<1xf32>
//       CHECK:   %[[I0:.+]] = vector.insert %[[E0]], %[[ZERO]] [0] : f32 into vector<4xf32>
//       CHECK:   %[[E1:.+]] = vector.extract %[[R1]][0] : f32 from vector<1xf32>
//       CHECK:   %[[I1:.+]] = vector.insert %[[E1]], %[[I0]] [1] : f32 into vector<4xf32>
//       CHECK:   %[[E2:.+]] = vector.extract %[[R2]][0] : f32 from vector<1xf32>
//       CHECK:   %[[I2:.+]] = vector.insert %[[E2]], %[[I1]] [2] : f32 into vector<4xf32>
//       CHECK:   %[[E3:.+]] = vector.extract %[[R3]][0] : f32 from vector<1xf32>
//       CHECK:   %[[I3:.+]] = vector.insert %[[E3]], %[[I2]] [3] : f32 into vector<4xf32>

//       CHECK:   %[[W:.+]] = vector.transfer_write %[[I3]], %{{.+}}
//       CHECK:   return %[[W]] : tensor<1x1x4xf32>

// -----

func.func @transpose_add(%lhs: tensor<4x2xf32>, %rhs: tensor<2xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x2xf32>, tensor<2xf32>)
    outs(%init : tensor<2x4xf32>) {
        ^bb0(%a : f32, %b : f32, %c : f32):
        %add = arith.addf %a, %b : f32
        linalg.yield %add : f32
  } -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @transpose_add
//  CHECK-SAME: (%[[LHS:.+]]: tensor<4x2xf32>, %[[RHS:.+]]: tensor<2xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index

//   CHECK-DAG:   %[[OINIT:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>

//       CHECK:   %[[LHS0:.+]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]]]{{.+}} : tensor<4x2xf32>, vector<2xf32>
//       CHECK:   %[[LHS1:.+]] = vector.transfer_read %[[LHS]][%[[C1]], %[[C0]]]{{.+}} : tensor<4x2xf32>, vector<2xf32>
//       CHECK:   %[[LHS2:.+]] = vector.transfer_read %[[LHS]][%[[C2]], %[[C0]]]{{.+}} : tensor<4x2xf32>, vector<2xf32>
//       CHECK:   %[[LHS3:.+]] = vector.transfer_read %[[LHS]][%[[C3]], %[[C0]]]{{.+}} : tensor<4x2xf32>, vector<2xf32>
//       CHECK:   %[[RHS0:.+]] = vector.transfer_read %[[RHS]][%[[C0]]]{{.+}} : tensor<2xf32>, vector<2xf32>

//       CHECK:   %[[ADD0:.+]] = arith.addf %[[LHS0]], %[[RHS0]]
//       CHECK:   %[[ADD1:.+]] = arith.addf %[[LHS1]], %[[RHS0]]
//       CHECK:   %[[ADD2:.+]] = arith.addf %[[LHS2]], %[[RHS0]]
//       CHECK:   %[[ADD3:.+]] = arith.addf %[[LHS3]], %[[RHS0]]

//       CHECK:   %[[E0:.+]] = vector.extract %[[ADD0]][0]
//       CHECK:   %[[I0:.+]] = vector.insert %[[E0]], %[[OINIT]] [0]
//       CHECK:   %[[E1:.+]] = vector.extract %[[ADD1]][0]
//       CHECK:   %[[I1:.+]] = vector.insert %[[E1]], %[[I0]] [1]
//       CHECK:   %[[E2:.+]] = vector.extract %[[ADD2]][0]
//       CHECK:   %[[I2:.+]] = vector.insert %[[E2]], %[[I1]] [2]
//       CHECK:   %[[E3:.+]] = vector.extract %[[ADD3]][0]
//       CHECK:   %[[I3:.+]] = vector.insert %[[E3]], %[[I2]] [3]
//       CHECK:   %[[E4:.+]] = vector.extract %[[ADD0]][1]
//       CHECK:   %[[I4:.+]] = vector.insert %[[E4]], %[[OINIT]] [0]
//       CHECK:   %[[E5:.+]] = vector.extract %[[ADD1]][1]
//       CHECK:   %[[I5:.+]] = vector.insert %[[E5]], %[[I4]] [1]
//       CHECK:   %[[E6:.+]] = vector.extract %[[ADD2]][1]
//       CHECK:   %[[I6:.+]] = vector.insert %[[E6]], %[[I5]] [2]
//       CHECK:   %[[E7:.+]] = vector.extract %[[ADD3]][1]
//       CHECK:   %[[I7:.+]] = vector.insert %[[E7]], %[[I6]] [3]

//       CHECK:   %[[W0:.+]] = vector.transfer_write %[[I3]], %{{.+}}[%[[C0]], %[[C0]]]
//       CHECK:   %[[W1:.+]] = vector.transfer_write %[[I7]], %[[W0]][%[[C1]], %[[C0]]]
//       CHECK:   return %[[W1]]

// -----

func.func @transpose_nd(%input: tensor<2x4x2x1x1xf32>) -> tensor<2x2x1x1x4xf32> {
  %init = tensor.empty() : tensor<2x2x1x1x4xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x4x2x1x1xf32>) outs(%init : tensor<2x2x1x1x4xf32>) {
  ^bb0(%arg0 : f32, %arg1 : f32):
    linalg.yield %arg0 : f32
  } -> tensor<2x2x1x1x4xf32>
  return %0: tensor<2x2x1x1x4xf32>
}

//    CHECK-LABEL: func @transpose_nd
//     CHECK-SAME: (%[[INPUT:.+]]: tensor<2x4x2x1x1xf32>)
// CHECK-COUNT-16:   vector.transfer_read %[[INPUT]]{{.+}} : tensor<2x4x2x1x1xf32>, vector<1xf32>
// CHECK-COUNT-16:   vector.insert {{.+}} : f32 into vector<4xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} : vector<4xf32>, tensor<2x2x1x1x4xf32>
