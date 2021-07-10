// RUN: iree-opt -iree-linalg-ext-tile -split-input-file %s | IreeFileCheck %s

func @scatter_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg_ext.scatter
    {__internal_linalg_transform__ = "scatter_tiling_input"}
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func @scatter_tiling(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[UPDATES]], %[[C0]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[ORIGINAL]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//   CHECK-DAG:     %[[D1:.+]] = tensor.dim %[[UPDATES]], %[[C1]]
//       CHECK:     %[[UPDATE_SLICE:.+]] = tensor.extract_slice %[[UPDATES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[INDEX_SLICE:.+]] = tensor.extract_slice %[[INDICES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], 1]
//       CHECK:     %[[SCATTER_TILE:.+]] = linalg_ext.scatter
//  CHECK-SAME:         __internal_linalg_transform__ = "scatter_tiling_output"
//  CHECK-SAME:         ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:         outs(%[[INIT]]
//   CHECK-DAG:     %[[SLICE_D0:.+]] = tensor.dim %[[SCATTER_TILE]], %[[C0]]
//   CHECK-DAG:     %[[SLICE_D1:.+]] = tensor.dim %[[SCATTER_TILE]], %[[C1]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SCATTER_TILE]] into %[[INIT]][0, 0]
//  CHECK-SAME:         [%[[SLICE_D0]], %[[SLICE_D1]]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// -----

func @scatter_tiling_memref(
    %original: memref<?x?xf32>, %indices: memref<?x1xi32>,
    %update : memref<?x?xf32>) {
  linalg_ext.scatter
    {__internal_linalg_transform__ = "scatter_tiling_input"}
    ins(%update, %indices : memref<?x?xf32>, memref<?x1xi32>)
    outs(%original : memref<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    }
  return
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func @scatter_tiling_memref(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[UPDATES]], %[[C0]]
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZE]]
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//   CHECK-DAG:     %[[D1:.+]] = memref.dim %[[UPDATES]], %[[C1]]
//       CHECK:     %[[UPDATE_SLICE:.+]] = memref.subview %[[UPDATES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[INDEX_SLICE:.+]] = memref.subview %[[INDICES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], 1]
//       CHECK:     linalg_ext.scatter
//  CHECK-SAME:         __internal_linalg_transform__ = "scatter_tiling_output"
//  CHECK-SAME:         ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:         outs(%[[ORIGINAL]]

// -----

func @scatter_tiling_distribution(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg_ext.scatter
    {__internal_linalg_transform__ = "scatter_distribute_input"}
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func @scatter_tiling_distribution(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[UPDATES]], %[[C0]]
//   CHECK-DAG:   %[[ID:.+]] = flow.dispatch.workgroup.id[0]
//   CHECK-DAG:   %[[COUNT:.+]] = flow.dispatch.workgroup.count[0]
//   CHECK-DAG:   %[[OFFSET:.+]] = affine.apply #[[MAP0]]()[%[[ID]]]
//   CHECK-DAG:   %[[STEP:.+]] = affine.apply #[[MAP0]]()[%[[COUNT]]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[OFFSET]] to %[[D0]] step %[[STEP]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[ORIGINAL]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP1]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//   CHECK-DAG:     %[[D1:.+]] = tensor.dim %[[UPDATES]], %[[C1]]
//       CHECK:     %[[UPDATE_SLICE:.+]] = tensor.extract_slice %[[UPDATES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[INDEX_SLICE:.+]] = tensor.extract_slice %[[INDICES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], 1]
//       CHECK:     %[[SCATTER_TILE:.+]] = linalg_ext.scatter
//  CHECK-SAME:         __internal_linalg_transform__ = "scatter_distribute_output"
//  CHECK-SAME:         ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:         outs(%[[INIT]]
//   CHECK-DAG:     %[[SLICE_D0:.+]] = tensor.dim %[[SCATTER_TILE]], %[[C0]]
//   CHECK-DAG:     %[[SLICE_D1:.+]] = tensor.dim %[[SCATTER_TILE]], %[[C1]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SCATTER_TILE]] into %[[INIT]][0, 0]
//  CHECK-SAME:         [%[[SLICE_D0]], %[[SLICE_D1]]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// -----

func @scatter_no_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg_ext.scatter
    {__internal_linalg_transform__ = "scatter_no_tiling_input"}
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//       CHECK: func @scatter_no_tiling
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = linalg_ext.scatter
//  CHECK-SAME:       __internal_linalg_transform__ = "scatter_no_tiling_output"
//  CHECK-SAME:       ins(%[[UPDATES]], %[[INDICES]]
//  CHECK-SAME:       outs(%[[ORIGINAL]]
//       CHECK:   return %[[RESULT]]
