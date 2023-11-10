// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-generic-vectorization,iree-spirv-initial-vector-lowering,iree-codegen-hoist-redundant-vector-transfers,iree-spirv-final-vector-lowering))' \
// RUN:   %s | FileCheck %s

func.func @reduce_outmost_dim(%input: tensor<4x1x4xf32>, %init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %f0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%input : tensor<4x1x4xf32>) outs(%init : tensor<1x4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    linalg.yield %add : f32
  } -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func @reduce_outmost_dim
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<4x1x4xf32>, %[[INIT:.+]]: tensor<1x4xf32>)

//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index

//       CHECK:   %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]] {in_bounds = [true]} : tensor<4x1x4xf32>, vector<4xf32>
//       CHECK:   %[[V1:.+]] = vector.transfer_read %[[INPUT]][%[[C1]], %[[C0]], %[[C0]]], %[[F0]] {in_bounds = [true]} : tensor<4x1x4xf32>, vector<4xf32>
//       CHECK:   %[[V2:.+]] = vector.transfer_read %[[INPUT]][%[[C2]], %[[C0]], %[[C0]]], %[[F0]] {in_bounds = [true]} : tensor<4x1x4xf32>, vector<4xf32>
//       CHECK:   %[[V3:.+]] = vector.transfer_read %[[INPUT]][%[[C3]], %[[C0]], %[[C0]]], %[[F0]] {in_bounds = [true]} : tensor<4x1x4xf32>, vector<4xf32>
//       CHECK:   %[[VI:.+]] = vector.transfer_read %[[INIT]][%[[C0]], %[[C0]]], %[[F0]] {in_bounds = [true]} : tensor<1x4xf32>, vector<4xf32>

//       CHECK:   %[[ADD0:.+]] = arith.addf %[[V0]], %[[VI]] : vector<4xf32>
//       CHECK:   %[[ADD1:.+]] = arith.addf %[[V1]], %[[ADD0]] : vector<4xf32>
//       CHECK:   %[[ADD2:.+]] = arith.addf %[[V2]], %[[ADD1]] : vector<4xf32>
//       CHECK:   %[[ADD3:.+]] = arith.addf %[[V3]], %[[ADD2]] : vector<4xf32>
//       CHECK:   %[[W:.+]] = vector.transfer_write %[[ADD3]], %[[INIT]][%[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, tensor<1x4xf32>
//       CHECK:   return %[[W]]

// -----

func.func @reduce_two_dims(%input: tensor<2x3x4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  %f0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>],
    iterator_types = ["parallel", "reduction", "reduction"]
  } ins(%input : tensor<2x3x4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

//   CHECK-LABEL: func @reduce_two_dims
//    CHECK-SAME: (%[[INPUT:.+]]: tensor<2x3x4xf32>, %[[INIT:.+]]: tensor<4xf32>)

// CHECK-COUNT-6:   vector.transfer_read %[[INPUT]]{{.+}} : tensor<2x3x4xf32>, vector<4xf32>
//     CHECK-NOT:   vector<{{.+}}x4xf32>
//     CHECK-NOT:   vector.insert
//     CHECK-NOT:   vector.broadcast
//     CHECK-NOT:   vector.transpose
//         CHECK:   vector.transfer_read %[[INIT]]{{.+}} : tensor<4xf32>, vector<4xf32>
// CHECK-COUNT-6:   arith.addf {{.+}} : vector<4xf32>
//         CHECK:   vector.transfer_write %{{.+}}, %[[INIT]]{{.+}} : vector<4xf32>, tensor<4xf32>

// -----

#map0 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>

func.func @reduce_multi_inputs_no_contraction(%a: tensor<2x3x4xf32>, %b: tensor<2x3x4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "reduction", "reduction"]
  } ins(%a, %b : tensor<2x3x4xf32>, tensor<2x3x4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %max = arith.maximumf %arg0, %arg1 : f32
    %add = arith.addf %max, %arg2 : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>
  return %0: tensor<4xf32>
}

//   CHECK-LABEL: func @reduce_multi_inputs_no_contraction
//    CHECK-SAME: (%[[A:.+]]: tensor<2x3x4xf32>, %[[B:.+]]: tensor<2x3x4xf32>, %[[INIT:.+]]: tensor<4xf32>)

// CHECK-COUNT-6:   vector.transfer_read %[[A]]{{.+}} : tensor<2x3x4xf32>, vector<4xf32>
//     CHECK-NOT:   vector<{{.+}}x4xf32>
//     CHECK-NOT:   vector.insert
//     CHECK-NOT:   vector.broadcast
//     CHECK-NOT:   vector.transpose
// CHECK-COUNT-6:   vector.transfer_read %[[B]]{{.+}} : tensor<2x3x4xf32>, vector<4xf32>
//     CHECK-NOT:   vector<{{.+}}x4xf32>
//     CHECK-NOT:   vector.insert
//     CHECK-NOT:   vector.broadcast
//     CHECK-NOT:   vector.transpose
//         CHECK:   vector.transfer_read %[[INIT]]{{.+}} : tensor<4xf32>, vector<4xf32>
// CHECK-COUNT-6:   arith.maximumf {{.+}} : vector<4xf32>
// CHECK-COUNT-6:   arith.addf {{.+}} : vector<4xf32>
//         CHECK:   vector.transfer_write %{{.+}}, %[[INIT]]{{.+}} : vector<4xf32>, tensor<4xf32>


// -----

#map0 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>

func.func @reduce_multi_inputs_contraction(%a: tensor<2x3x4xf32>, %b: tensor<2x3x4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map0, #map1],
    iterator_types = ["parallel", "reduction", "reduction"]
  } ins(%a, %b : tensor<2x3x4xf32>, tensor<2x3x4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %max = arith.mulf %arg0, %arg1 : f32
    %add = arith.addf %max, %arg2 : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>
  return %0: tensor<4xf32>
}

//   CHECK-LABEL: func @reduce_multi_inputs_contraction
//    CHECK-SAME: (%[[A:.+]]: tensor<2x3x4xf32>, %[[B:.+]]: tensor<2x3x4xf32>, %[[INIT:.+]]: tensor<4xf32>)

// CHECK-COUNT-6:   vector.transfer_read %[[A]]{{.+}} : tensor<2x3x4xf32>, vector<4xf32>
//     CHECK-NOT:   vector<{{.+}}x4xf32>
//     CHECK-NOT:   vector.insert
//     CHECK-NOT:   vector.broadcast
//     CHECK-NOT:   vector.transpose
// CHECK-COUNT-6:   vector.transfer_read %[[B]]{{.+}} : tensor<2x3x4xf32>, vector<4xf32>
//     CHECK-NOT:   vector<{{.+}}x4xf32>
//     CHECK-NOT:   vector.insert
//     CHECK-NOT:   vector.broadcast
//     CHECK-NOT:   vector.transpose
//         CHECK:   vector.transfer_read %[[INIT]]{{.+}} : tensor<4xf32>, vector<4xf32>
// CHECK-COUNT-6:   vector.fma {{.+}} : vector<4xf32>
//         CHECK:   vector.transfer_write %{{.+}}, %[[INIT]]{{.+}} : vector<4xf32>, tensor<4xf32>

// -----

func.func @reduce_innermost_dim_contraction(%a: tensor<4x12xf32>, %b: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%a, %b : tensor<4x12xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sub = arith.subf %arg0, %arg1 : f32
    %mul = arith.mulf %sub, %sub : f32
    %add = arith.addf %mul, %arg2 : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>
  return %0: tensor<4xf32>
}


//    CHECK-LABEL: func @reduce_innermost_dim_contraction
//     CHECK-SAME: (%[[A:.+]]: tensor<4x12xf32>, %[[B:.+]]: tensor<4xf32>, %[[INIT:.+]]: tensor<4xf32>)

// CHECK-COUNT-12: vector.transfer_read %[[A]]{{.+}} : tensor<4x12xf32>, vector<4xf32>
//          CHECK: vector.transfer_read %[[B]]{{.+}} : tensor<4xf32>, vector<4xf32>
//          CHECK: vector.transfer_read %[[INIT]]{{.+}} : tensor<4xf32>, vector<4xf32>
// CHECK-COUNT-12: arith.subf {{.+}} : vector<4xf32>
// CHECK-COUNT-12: vector.fma {{.+}} : vector<4xf32>
//          CHECK: vector.transfer_write %{{.+}}, %[[INIT]]{{.+}} : vector<4xf32>, tensor<4xf32>

// -----

func.func @reduce_vector3(%input: tensor<4x3xf32>, %init: tensor<3xf32>) -> tensor<3xf32> {
  %f0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<4x3xf32>) outs(%init : tensor<3xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    linalg.yield %add : f32
  } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @reduce_vector3
// CHECK-COUNT-4: arith.addf {{.+}} : vector<3xf32>
