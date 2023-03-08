// RUN: iree-dialects-opt --iree-linalg-ext-tile --split-input-file --verify-diagnostics -cse %s | FileCheck  %s

func.func @scatter_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
    {__internal_linalg_transform__ = "tiling_input"}
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func.func @scatter_tiling(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[TILESIZEY:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[TILESIZEX:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[UPDATES]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[UPDATES]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZEY]]
//  CHECK-SAME:       iter_args(%[[INITY:.+]] = %[[ORIGINAL]])
//   CHECK-DAG:     %[[USED_TILESIZEY:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[TILESIZEY]], %[[D0]]]
//       CHECK:     %[[RESULT_INNER:.+]] = scf.for %[[IV1:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZEX]]
//  CHECK-SAME:         iter_args(%[[INITX:.+]] = %[[INITY]])
//       CHECK:       %[[USED_TILESIZEX:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[TILESIZEX]], %[[D1]]]
//       CHECK:       %[[UPDATE_SLICE:.+]] = tensor.extract_slice %[[UPDATES]][%[[IV0]], %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], %[[USED_TILESIZEX]]]
//       CHECK:       %[[INDEX_SLICE:.+]] = tensor.extract_slice %[[INDICES]][%[[IV0]], 0]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], 1]
//       CHECK:       %[[SCATTER_DIM:.+]] = tensor.dim %[[ORIGINAL]], %[[C0]]
//       CHECK:       %[[ORIGINAL_SLICE:.+]] = tensor.extract_slice %[[ORIGINAL]][0, %[[IV1]]]
//  CHECK-SAME:           [%[[SCATTER_DIM]], %[[USED_TILESIZEX]]]
//       CHECK:       %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:           __internal_linalg_transform__ = "tiling_output"
//  CHECK-SAME:           unique_indices(true)
//  CHECK-SAME:           ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:           outs(%[[ORIGINAL_SLICE]]
//       CHECK:       %[[YIELD:.+]] = tensor.insert_slice %[[SCATTER_TILE]] into %[[INITX]][0, %[[IV1]]]
//  CHECK-SAME:           [%[[SCATTER_DIM]], %[[USED_TILESIZEX]]]
//       CHECK:       scf.yield %[[YIELD]]
//       CHECK:     scf.yield %[[RESULT_INNER]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_tiling_memref(
    %original: memref<?x?xf32>, %indices: memref<?x1xi32>,
    %update : memref<?x?xf32>) {
  iree_linalg_ext.scatter
    {__internal_linalg_transform__ = "tiling_input"}
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : memref<?x?xf32>, memref<?x1xi32>)
    outs(%original : memref<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    }
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func.func @scatter_tiling_memref(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[TILESIZEY:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[TILESIZEX:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[UPDATES]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[UPDATES]], %[[C1]]
//       CHECK:   scf.for %[[IV0:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZEY]]
//   CHECK-DAG:     %[[USED_TILESIZEY:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[TILESIZEY]], %[[D0]]]
//       CHECK:     scf.for %[[IV1:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZEX]]
//   CHECK-DAG:       %[[USED_TILESIZEX:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[TILESIZEX]], %[[D1]]]
//       CHECK:       %[[UPDATE_SLICE:.+]] = memref.subview %[[UPDATES]][%[[IV0]], %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], %[[USED_TILESIZEX]]]
//       CHECK:       %[[INDEX_SLICE:.+]] = memref.subview %[[INDICES]][%[[IV0]], 0]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], 1]
//       CHECK:       %[[SCATTER_DIM:.+]] = memref.dim %[[ORIGINAL]], %[[C0]]
//       CHECK:       %[[ORIGINAL_SLICE:.+]] = memref.subview %[[ORIGINAL]][0, %[[IV1]]
//  CHECK-SAME:           [%[[SCATTER_DIM]], %[[USED_TILESIZEX]]]
//       CHECK:       iree_linalg_ext.scatter
//  CHECK-SAME:           __internal_linalg_transform__ = "tiling_output"
//  CHECK-SAME:           unique_indices(true)
//  CHECK-SAME:           ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:           outs(%[[ORIGINAL_SLICE]]

// -----

func.func @scatter_tiling_distribution(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
    {__internal_linalg_transform__ = "distribute_input"}
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func.func @scatter_tiling_distribution(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[UPDATES]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[UPDATES]], %[[C1]]
//   CHECK-DAG:   %[[ID:.+]] = iree_input.dispatch.workgroup.id[0]
//   CHECK-DAG:   %[[COUNT:.+]] = iree_input.dispatch.workgroup.count[0]
//   CHECK-DAG:   %[[OFFSET:.+]] = affine.apply #[[MAP0]]()[%[[ID]]]
//   CHECK-DAG:   %[[STEP:.+]] = affine.apply #[[MAP0]]()[%[[COUNT]]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[OFFSET]] to %[[D0]] step %[[STEP]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[ORIGINAL]])
//       CHECK:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP1]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//       CHECK:     %[[UPDATE_SLICE:.+]] = tensor.extract_slice %[[UPDATES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[INDEX_SLICE:.+]] = tensor.extract_slice %[[INDICES]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], 1]
//       CHECK:     %[[D2:.+]] = tensor.dim %[[ORIGINAL]], %[[C0]]
//       CHECK:     %[[ORIGINAL_SLICE:.+]] = tensor.extract_slice %[[ORIGINAL]][0, 0]
//  CHECK-SAME:         [%[[D2]], %[[D1]]]
//       CHECK:     %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:        __internal_linalg_transform__ = "distribute_output"
//  CHECK-SAME:        unique_indices(true)
//  CHECK-SAME:        ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:        outs(%[[ORIGINAL_SLICE]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SCATTER_TILE]] into %[[INIT]][0, 0]
//  CHECK-SAME:        [%[[D2]], %[[D1]]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_no_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
    {__internal_linalg_transform__ = "no_tiling_input"}
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//       CHECK: func.func @scatter_no_tiling
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       __internal_linalg_transform__ = "no_tiling_output"
//  CHECK-SAME:       unique_indices(true)
//  CHECK-SAME:       ins(%[[UPDATES]], %[[INDICES]]
//  CHECK-SAME:       outs(%[[ORIGINAL]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_repeated_indices_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
    {__internal_linalg_transform__ = "tiling_repeated_indices_scatter_input"}
    dimension_map = [0]
    unique_indices(false)
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func.func @scatter_repeated_indices_tiling
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[UPDATES]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[UPDATES]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[ITER:.+]] = %[[ORIGINAL]])
//       CHECK:     %[[SZ:.+]] = affine.min #[[MAP]](%[[I]])[%[[TILESIZE]], %[[D1]]]
//       CHECK:       %[[UPDATES_TILE:.+]] = tensor.extract_slice
//  CHECK-SAME:         %[[UPDATES]][0, %[[I]]] [%[[D0]], %[[SZ]]] [1, 1]
//       CHECK:       %[[INDICES_TILE:.+]] = tensor.extract_slice
//  CHECK-SAME:         %[[INDICES]][0, 0] [%[[D0]], 1] [1, 1]
//       CHECK:       %[[ORIGINAL_D0:.+]] = tensor.dim %[[ORIGINAL]], %[[C0]]
//       CHECK:       %[[ORIGINAL_TILE:.+]] = tensor.extract_slice
//  CHECK-SAME:         %[[ORIGINAL]][0, %[[I]]] [%[[ORIGINAL_D0]], %[[SZ]]] [1, 1]
//       CHECK:       %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:         __internal_linalg_transform__ = "tiling_repeated_indices_scatter_output"
//  CHECK-SAME:         unique_indices(false)
//  CHECK-SAME:         ins(%[[UPDATES_TILE]], %[[INDICES_TILE]]
//  CHECK-SAME:         outs(%[[ORIGINAL_TILE]]
//       CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SCATTER]] into
//  CHECK-SAME:         %[[ITER]][0, %[[I]]] [%[[ORIGINAL_D0]], %[[SZ]]] [1, 1]
//       CHECK:       scf.yield %[[RES]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_repeated_indices_no_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{unimplemented tiling of non-parallel loop iterator type}}
  %0 = iree_linalg_ext.scatter
    {__internal_linalg_transform__ = "tiling_input"}
    dimension_map = [0]
    unique_indices(false)
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @sort_1d(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = iree_linalg_ext.sort
       {__internal_linalg_transform__ = "outer_reduce_input"}
       dimension(0)
       outs(%arg0 : tensor<?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
         %0 = arith.cmpi sgt, %arg2, %arg3 : i32
         iree_linalg_ext.yield %0 : i1
       } -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
//      CHECK: func.func @sort_1d(
// CHECK-SAME:   %[[OPERAND:.+]]: tensor<?xi32>
//      CHECK:   %[[RESULT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:       {__internal_linalg_transform__ = "outer_reduce_output"}
// CHECK-SAME:       outs(%[[OPERAND]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @sort_2d(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = iree_linalg_ext.sort
       {__internal_linalg_transform__ = "inner_reduce_input"}
       dimension(1)
       outs(%arg0 : tensor<?x?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
         %0 = arith.cmpi sgt, %arg2, %arg3 : i32
         iree_linalg_ext.yield %0 : i1
       } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func.func @sort_2d(
//  CHECK-SAME:   %[[OPERAND:.+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[OPERAND]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//       CHECK:     %[[OPERAND_SLICE:.+]] = tensor.extract_slice %[[OPERAND]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[SORT_TILE:.+]] = iree_linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "inner_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND_SLICE]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SORT_TILE]] into %[[INIT]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @sort_2d_inner_parallel(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = iree_linalg_ext.sort
       {__internal_linalg_transform__ = "outer_reduce_input"}
       dimension(0)
       outs(%arg0 : tensor<?x?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
         %0 = arith.cmpi sgt, %arg2, %arg3 : i32
         iree_linalg_ext.yield %0 : i1
       } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func.func @sort_2d_inner_parallel(
//  CHECK-SAME:   %[[OPERAND:.+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[OPERAND]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D1]]]
//       CHECK:     %[[OPERAND_SLICE:.+]] = tensor.extract_slice %[[OPERAND]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     %[[SORT_TILE:.+]] = iree_linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "outer_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND_SLICE]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SORT_TILE]] into %[[INIT]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @sort_2d_multi_result(
    %arg0: tensor<?x?xi32>, %arg1: tensor<?x?xf32>)
    -> (tensor<?x?xi32>, tensor<?x?xf32>) {
  %0:2 = iree_linalg_ext.sort
       {__internal_linalg_transform__ = "inner_reduce_input"}
       dimension(1)
       outs(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xf32>) {
       ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
         %1 = arith.cmpf ogt, %arg4, %arg5 : f32
         iree_linalg_ext.yield %1 : i1
       } -> tensor<?x?xi32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xf32>
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func.func @sort_2d_multi_result(
//  CHECK-SAME:   %[[OPERAND1:.+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:.+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND1]], %[[C1]]
//       CHECK:   %[[RESULT:.+]]:2 = scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT1:.+]] = %[[OPERAND1]], %[[INIT2:.+]] = %[[OPERAND2]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//       CHECK:     %[[OPERAND1_SLICE:.+]] = tensor.extract_slice %[[OPERAND1]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[OPERAND2_SLICE:.+]] = tensor.extract_slice %[[OPERAND2]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[SORT_TILE:.+]]:2 = iree_linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "inner_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]
//       CHECK:     %[[YIELD1:.+]] = tensor.insert_slice %[[SORT_TILE]]#0 into %[[INIT1]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[YIELD2:.+]] = tensor.insert_slice %[[SORT_TILE]]#1 into %[[INIT2]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     scf.yield %[[YIELD1]], %[[YIELD2]]
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @sort_2d_multi_result_memref(
    %arg0: memref<?x?xi32>, %arg1: memref<?x?xf32>) {
  iree_linalg_ext.sort
     {__internal_linalg_transform__ = "outer_reduce_input"}
     dimension(0)
     outs(%arg0, %arg1 : memref<?x?xi32>, memref<?x?xf32>) {
     ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
       %0 = arith.cmpf ogt, %arg4, %arg5 : f32
       iree_linalg_ext.yield %0 : i1
     }
  return
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func.func @sort_2d_multi_result_memref(
//  CHECK-SAME:   %[[OPERAND1:.+]]: memref<?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:.+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[OPERAND1]], %[[C1]]
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZE]]
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D1]]]
//       CHECK:     %[[OPERAND1_SLICE:.+]] = memref.subview %[[OPERAND1]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     %[[OPERAND2_SLICE:.+]] = memref.subview %[[OPERAND2]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     iree_linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "outer_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]

// -----

func.func @sort_3d_multi_result_distribute(
  %arg0: tensor<?x?x?xi32>, %arg1 : tensor<?x?x?xf32>)
  -> (tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
  %0, %1 = iree_linalg_ext.sort
      {__internal_linalg_transform__ = "distribute_input"}
      dimension(1)
      outs(%arg0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %2 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %2 : i1
      } -> tensor<?x?x?xi32>, tensor<?x?x?xf32>
  return %0, %1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
//       CHECK: func.func @sort_3d_multi_result_distribute(
//  CHECK-SAME:   %[[OPERAND1:[a-zA-Z0-9_]+]]: tensor<?x?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//   CHECK-DAG:   %[[TILESIZE1:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[TILESIZE2:.+]] = arith.constant 30 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND1]], %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[OPERAND1]], %[[C2]]
//   CHECK-DAG:   %[[IDX:.+]] = iree_input.dispatch.workgroup.id[0]
//   CHECK-DAG:   %[[COUNTX:.+]] = iree_input.dispatch.workgroup.count[0]
//   CHECK-DAG:   %[[IDY:.+]] = iree_input.dispatch.workgroup.id[1]
//   CHECK-DAG:   %[[COUNTY:.+]] = iree_input.dispatch.workgroup.count[1]
//   CHECK-DAG:   %[[OFFSETY:.+]] = affine.apply #[[MAP0]]()[%[[IDY]]]
//   CHECK-DAG:   %[[STEPY:.+]] = affine.apply #[[MAP0]]()[%[[COUNTY]]]
//       CHECK:   %[[RESULT:.+]]:2 = scf.for %[[IV0:.+]] = %[[OFFSETY]] to %[[D0]] step %[[STEPY]]
//  CHECK-SAME:       iter_args(%[[INIT1:.+]] = %[[OPERAND1]], %[[INIT2:.+]] = %[[OPERAND2]])
//   CHECK-DAG:     %[[USED_TILESIZE1:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[TILESIZE1]], %[[D0]]]
//   CHECK-DAG:     %[[OFFSETX:.+]] = affine.apply #[[MAP2]]()[%[[IDX]]]
//   CHECK-DAG:     %[[STEPX:.+]] = affine.apply #[[MAP2]]()[%[[COUNTX]]]
//       CHECK:     %[[RESULT_INNER:.+]]:2 = scf.for %[[IV1:.+]] = %[[OFFSETX]] to %[[D2]] step %[[STEPX]]
//  CHECK-SAME:         iter_args(%[[INIT3:.+]] = %[[INIT1]], %[[INIT4:.+]] = %[[INIT2]])
//   CHECK-DAG:       %[[USED_TILESIZE2:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[TILESIZE2]], %[[D2]]]
//       CHECK:       %[[OPERAND1_SLICE:.+]] = tensor.extract_slice %[[OPERAND1]][%[[IV0]], 0, %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZE1]], %[[D1]], %[[USED_TILESIZE2]]]
//       CHECK:       %[[OPERAND2_SLICE:.+]] = tensor.extract_slice %[[OPERAND2]][%[[IV0]], 0, %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZE1]], %[[D1]], %[[USED_TILESIZE2]]]
//       CHECK:       %[[SORT_SLICE:.+]]:2 = iree_linalg_ext.sort
//  CHECK-SAME:           __internal_linalg_transform__ = "distribute_output"
//  CHECK-SAME:           outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]
//       CHECK:       %[[YIELD1:.+]] = tensor.insert_slice %[[SORT_SLICE]]#0
//  CHECK-SAME:           into %[[INIT3]][%[[IV0]], 0, %[[IV1]]]
//       CHECK:       %[[YIELD2:.+]] = tensor.insert_slice %[[SORT_SLICE]]#1
//  CHECK-SAME:           into %[[INIT4]][%[[IV0]], 0, %[[IV1]]]
//       CHECK:       scf.yield %[[YIELD1]], %[[YIELD2]]
//       CHECK:     scf.yield %[[RESULT_INNER]]#0, %[[RESULT_INNER]]#1
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @sort_3d_multi_result_distribute_memref(
  %arg0: memref<?x?x?xi32>, %arg1 : memref<?x?x?xf32>) {
  iree_linalg_ext.sort
      {__internal_linalg_transform__ = "distribute_input"}
      dimension(1)
      outs(%arg0, %arg1 : memref<?x?x?xi32>, memref<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %0 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %0 : i1
      }
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
//       CHECK: func.func @sort_3d_multi_result_distribute_memref(
//  CHECK-SAME:   %[[OPERAND1:[a-zA-Z0-9_]+]]: memref<?x?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:[a-zA-Z0-9_]+]]: memref<?x?x?xf32>
//   CHECK-DAG:   %[[TILESIZE1:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[TILESIZE2:.+]] = arith.constant 30 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[OPERAND1]], %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = memref.dim %[[OPERAND1]], %[[C2]]
//   CHECK-DAG:   %[[IDX:.+]] = iree_input.dispatch.workgroup.id[0]
//   CHECK-DAG:   %[[COUNTX:.+]] = iree_input.dispatch.workgroup.count[0]
//   CHECK-DAG:   %[[IDY:.+]] = iree_input.dispatch.workgroup.id[1]
//   CHECK-DAG:   %[[COUNTY:.+]] = iree_input.dispatch.workgroup.count[1]
//   CHECK-DAG:   %[[OFFSETY:.+]] = affine.apply #[[MAP0]]()[%[[IDY]]]
//   CHECK-DAG:   %[[STEPY:.+]] = affine.apply #[[MAP0]]()[%[[COUNTY]]]
//       CHECK:   scf.for %[[IV0:.+]] = %[[OFFSETY]] to %[[D0]] step %[[STEPY]]
//   CHECK-DAG:     %[[USED_TILESIZE1:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[TILESIZE1]], %[[D0]]]
//   CHECK-DAG:     %[[OFFSETX:.+]] = affine.apply #[[MAP2]]()[%[[IDX]]]
//   CHECK-DAG:     %[[STEPX:.+]] = affine.apply #[[MAP2]]()[%[[COUNTX]]]
//       CHECK:     scf.for %[[IV1:.+]] = %[[OFFSETX]] to %[[D2]] step %[[STEPX]]
//   CHECK-DAG:       %[[USED_TILESIZE2:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[TILESIZE2]], %[[D2]]]
//       CHECK:       %[[OPERAND1_SLICE:.+]] = memref.subview %[[OPERAND1]][%[[IV0]], 0, %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZE1]], %[[D1]], %[[USED_TILESIZE2]]]
//       CHECK:       %[[OPERAND2_SLICE:.+]] = memref.subview %[[OPERAND2]][%[[IV0]], 0, %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZE1]], %[[D1]], %[[USED_TILESIZE2]]]
//       CHECK:       iree_linalg_ext.sort
//  CHECK-SAME:           __internal_linalg_transform__ = "distribute_output"
//  CHECK-SAME:           outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]

// -----

func.func @fft_1d_stage_5(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>,
    %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  %cst1 = arith.constant 5 : index
  %0:2 = iree_linalg_ext.fft
  {__internal_linalg_transform__ = "tiling_1d_stage5_fft_input"}
    ins(%cst1, %arg2, %arg3: index, tensor<16xf32>, tensor<16xf32>)
    outs(%arg0, %arg1: tensor<1024xf32>, tensor<1024xf32>)
  : tensor<1024xf32>, tensor<1024xf32>
  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @fft_1d_stage_5(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[RES:.+]]:2 = scf.for %[[I:.+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG5:.+]] = %[[ARG0]], %[[ARG6:.+]] = %[[ARG1]])
// CHECK-SAME:       -> (tensor<1024xf32>, tensor<1024xf32>) {
// CHECK:          %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C32]], %[[C1024]]]
// CHECK:          %[[SLICE1:.+]] = tensor.extract_slice %[[ARG0]][%[[I]]] [%[[SIZE]]] [1] : tensor<1024xf32> to tensor<?xf32>
// CHECK:          %[[SLICE2:.+]] = tensor.extract_slice %[[ARG1]][%[[I]]] [%[[SIZE]]] [1] : tensor<1024xf32> to tensor<?xf32>
// CHECK:          %[[FFT:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:       {__internal_linalg_transform__ = "tiling_1d_stage5_fft_output"}
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME:       outs(%[[SLICE1]], %[[SLICE2]] : tensor<?xf32>, tensor<?xf32>)
// CHECK:          %[[INSERT1:.+]] = tensor.insert_slice %[[FFT]]#0 into %[[ARG5]][%[[I]]] [%[[SIZE]]] [1] : tensor<?xf32> into tensor<1024xf32>
// CHECK:          %[[INSERT2:.+]] = tensor.insert_slice %[[FFT]]#1 into %[[ARG6]][%[[I]]] [%[[SIZE]]] [1] : tensor<?xf32> into tensor<1024xf32>
// CHECK:          scf.yield %[[INSERT1]], %[[INSERT2]]
// CHECK:        return %[[RES]]#0, %[[RES]]#1 : tensor<1024xf32>, tensor<1024xf32>

// -----

func.func @fft_2d_stage_5(%arg0: tensor<3x1024xf32>, %arg1: tensor<3x1024xf32>,
    %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
  %cst1 = arith.constant 5 : index
  %0:2 = iree_linalg_ext.fft
  {__internal_linalg_transform__ = "tiling_2d_stage5_fft_input"}
    ins(%cst1, %arg2, %arg3: index, tensor<16xf32>, tensor<16xf32>)
    outs(%arg0, %arg1: tensor<3x1024xf32>, tensor<3x1024xf32>)
  : tensor<3x1024xf32>, tensor<3x1024xf32>
  return %0#0, %0#1 : tensor<3x1024xf32>, tensor<3x1024xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @fft_2d_stage_5(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:    %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[RES:.+]]:2 = scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C10]]
// CHECK-SAME:       iter_args(%[[ARG5:.+]] = %[[ARG0]], %[[ARG6:.+]] = %[[ARG1]])
// CHECK-SAME:       -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
// CHECK:          %[[SZ1:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C10]], %[[C3]]]
// CHECK:          %{{.+}} = scf.for %[[J:.+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:         iter_args(%[[ARG8:.+]] = %[[ARG5]], %[[ARG9:.+]] = %[[ARG6]]) -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
// CHECK:            %[[SZ2:.+]] = affine.min #[[MAP1]](%[[J]])[%[[C32]], %[[C1024]]]
// CHECK:            %[[SLICE1:.+]] = tensor.extract_slice %[[ARG0]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:            %[[SLICE2:.+]] = tensor.extract_slice %[[ARG1]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:          %[[FFT:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:       {__internal_linalg_transform__ = "tiling_2d_stage5_fft_output"}
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME:       outs(%[[SLICE1]], %[[SLICE2]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK:          %[[INSERT1:.+]] = tensor.insert_slice %[[FFT]]#0 into %[[ARG8]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:          %[[INSERT2:.+]] = tensor.insert_slice %[[FFT]]#1 into %[[ARG9]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:          scf.yield %[[INSERT1]], %[[INSERT2]] : tensor<3x1024xf32>, tensor<3x1024xf32>

// -----

func.func @fft_1d_stage_5_memref(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>,
    %arg2: memref<16xf32>, %arg3: memref<16xf32>) {
  %cst1 = arith.constant 5 : index
  iree_linalg_ext.fft
  {__internal_linalg_transform__ = "tiling_1d_stage5_fft_input"}
    ins(%cst1, %arg2, %arg3: index, memref<16xf32>, memref<16xf32>)
    outs(%arg0, %arg1: memref<1024xf32>, memref<1024xf32>)
  return
}
// CHECK:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @fft_1d_stage_5_memref(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        scf.for %[[I:.+]] = %[[C0]] to %[[C1024]] step %[[C32]] {
// CHECK:          %[[SZ:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C32]], %[[C1024]]]
// CHECK:          %[[SUB1:.+]] = memref.subview %[[ARG0]][%[[I]]] [%[[SZ]]] [1] : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:          %[[SUB2:.+]] = memref.subview %[[ARG1]][%[[I]]] [%[[SZ]]] [1] : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:          iree_linalg_ext.fft
// CHECK-SAME:       {__internal_linalg_transform__ = "tiling_1d_stage5_fft_output"}
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, memref<16xf32>, memref<16xf32>)
// CHECK-SAME:       outs(%[[SUB1]], %[[SUB2]] : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>)

// -----

func.func @reverse_memref(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  iree_linalg_ext.reverse
    {__internal_linalg_transform__ = "tiling_input"}
    dimensions(dense<0> : tensor<1xi64>)
    ins(%arg0: memref<?xi32>)
    outs(%arg1: memref<?xi32>)
  return
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s0 - s1 - s2)>
// CHECK:      func.func @reverse_memref(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xi32>
// CHECK:        scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C10]] {
// CHECK-DAG:      %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C10]], %[[D0]]]
// CHECK-DAG:      %[[IDX:.+]] = affine.apply #[[MAP2]]()[%[[D0]], %[[I]], %[[SIZE]]]
// CHECK-DAG:      %[[SUB_IN:.+]] =  memref.subview %[[ARG0]][%[[I]]] [%[[SIZE]]] [1]
// CHECK-DAG:      %[[SUB_OUT:.+]] = memref.subview %[[ARG1]][%[[IDX]]] [%[[SIZE]]] [1]
// CHECK:          iree_linalg_ext.reverse
// CHECK-SAME:       {__internal_linalg_transform__ = "tiling_output"}
// CHECK-SAME:       dimensions(dense<0> : tensor<1xi64>)
// CHECK-SAME:       ins(%[[SUB_IN]]
// CHECK-SAME:       outs(%[[SUB_OUT]]

// -----

func.func @reverse_tensor_multi_dim(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %0 = iree_linalg_ext.reverse
         {__internal_linalg_transform__ = "tiling_input"}
         dimensions(dense<[0, 1]> : tensor<2xi64>)
         ins(%arg0: tensor<?x?xi32>)
         outs(%init: tensor<?x?xi32>) : tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s0 - s1 - s2)>
// CHECK:      func.func @reverse_tensor_multi_dim(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[C20:.+]] = arith.constant 20 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xi32>
// CHECK-DAG:    %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xi32>
// CHECK:        %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xi32>
// CHECK:        %[[RES:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C10]]
// CHECK-SAME:     iter_args(%[[INIT2:.+]] = %[[INIT]]) -> (tensor<?x?xi32>) {
// CHECK:          %[[SIZE_I:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C10]], %[[D0]]]
// CHECK:          %[[RES2:.+]] = scf.for %[[J:.+]] = %[[C0]] to %[[D1]] step %[[C20]]
// CHECK-SAME:       iter_args(%[[INIT3:.+]] = %[[INIT2]]) -> (tensor<?x?xi32>) {
// CHECK-DAG:        %[[SIZE_J:.+]] = affine.min #[[MAP1]](%[[J]])[%[[C20]], %[[D1]]]
// CHECK-DAG:        %[[IDX0:.+]] = affine.apply #[[MAP2]]()[%[[D0]], %[[I]], %[[SIZE_I]]]
// CHECK-DAG:        %[[IDX1:.+]] = affine.apply #[[MAP2]]()[%[[D1]], %[[J]], %[[SIZE_J]]]
// CHECK:            %[[SUB_IN:.+]] = tensor.extract_slice
// CHECK-SAME:         %[[ARG0]][%[[I]], %[[J]]] [%[[SIZE_I]], %[[SIZE_J]]] [1, 1]
// CHECK:            %[[SUB_INIT:.+]] = tensor.extract_slice
// CHECK-SAME:         %[[INIT]][%[[IDX0]], %[[IDX1]]] [%[[SIZE_I]], %[[SIZE_J]]] [1, 1]
// CHECK:            %[[REV:.+]] = iree_linalg_ext.reverse
// CHECK-SAME:          {__internal_linalg_transform__ = "tiling_output"}
// CHECK-SAME:          dimensions(dense<[0, 1]> : tensor<2xi64>)
// CHECK-SAME:          ins(%[[SUB_IN]]
// CHECK-SAME:          outs(%[[SUB_INIT]]
// CHECK:            %[[RES3:.+]] = tensor.insert_slice %[[REV]] into
// CHECK-SAME:         %[[INIT3]][%[[IDX0]], %[[IDX1]]] [%[[SIZE_I]], %[[SIZE_J]]] [1, 1]
// CHECK:            scf.yield %[[RES3]]
// CHECK:          scf.yield %[[RES2]]
// CHECK:        return %[[RES]]

// -----

func.func @scan_1d(%0: tensor<128xi32>) -> tensor<128xi32> {
  %c0 = tensor.empty() : tensor<i32>
  %1 = tensor.empty() : tensor<128xi32>
  %2:2 = iree_linalg_ext.scan
    {__internal_linalg_transform__ = "outer_reduce_input"}
    dimension(0) inclusive(true)
    ins(%0 : tensor<128xi32>) outs(%1, %c0 : tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %2#0 : tensor<128xi32>
}
//      CHECK: func.func @scan_1d(
// CHECK-SAME:   %[[OPERAND:.+]]: tensor<128xi32>
//      CHECK:   %[[ACC:.+]] = tensor.empty() : tensor<i32>
//      CHECK:   %[[OUTPUT:.+]] = tensor.empty() : tensor<128xi32>
//      CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.scan
// CHECK-SAME:           __internal_linalg_transform__ = "outer_reduce_output"
// CHECK-SAME:       ins(%[[OPERAND]] :
// CHECK-SAME:       outs(%[[OUTPUT]], %[[ACC]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @scan_2d(%0: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %c0 = tensor.empty() : tensor<32xi32>
  %1 = tensor.empty() : tensor<16x32xi32>
  %2:2 = iree_linalg_ext.scan
    {__internal_linalg_transform__ = "outer_reduce_input"}
    dimension(0) inclusive(true)
    ins(%0 : tensor<16x32xi32>) outs(%1, %c0 : tensor<16x32xi32>, tensor<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  } -> tensor<16x32xi32>, tensor<32xi32>
  return %2#0 : tensor<16x32xi32>
}
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//      CHECK:  func.func @scan_2d(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
//      CHECK:    %[[C0:.+]] = arith.constant 0 : index
//      CHECK:    %[[C16:.+]] = arith.constant 16 : index
//      CHECK:    %[[C32:.+]] = arith.constant 32 : index
//      CHECK:    %[[C20:.+]] = arith.constant 20 : index
//      CHECK:    %[[ACC:.+]] = tensor.empty() : tensor<32xi32>
//      CHECK:    %[[OUTPUT:.+]] = tensor.empty() : tensor<16x32xi32>
//      CHECK:    %[[RESULT:.+]]:2 = scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C20]]
// CHECK-SAME:      iter_args(%[[ARG2:.+]] = %[[OUTPUT]], %[[ARG3:.+]] = %[[ACC]])
//      CHECK:      %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C20]], %[[C32]]]
//      CHECK:      %[[UPDATE_SLICE_IN:.+]] = tensor.extract_slice %[[ARG0]][0, %[[I]]] [%[[C16]], %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_OUT:.+]] = tensor.extract_slice %[[OUTPUT]][0, %[[I]]] [%[[C16]], %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_ACC:.+]] = tensor.extract_slice %[[ACC]][%[[I]]] [%[[SIZE]]]
//      CHECK:      %[[SCAN_TILE:.+]]:2 = iree_linalg_ext.scan
// CHECK-SAME:       {__internal_linalg_transform__ = "outer_reduce_output"}
// CHECK-SAME:       dimension(0) inclusive(true)
// CHECK-SAME:       ins(%[[UPDATE_SLICE_IN]]
// CHECK-SAME:       outs(%[[UPDATE_SLICE_OUT]], %[[UPDATE_SLICE_ACC]]
//      CHECK:       %[[YIELD:.+]] = tensor.insert_slice %[[SCAN_TILE]]#0 into %[[ARG2]][0, %[[I]]]
// CHECK-SAME:           [%[[C16]], %[[SIZE]]]
//      CHECK:       %[[ACC_YIELD:.+]] = tensor.insert_slice %[[SCAN_TILE]]#1 into %[[ARG3]][%[[I]]]
// CHECK-SAME:           [%[[SIZE]]]
//      CHECK:       scf.yield %[[YIELD]], %[[ACC_YIELD]] : tensor<16x32xi32>, tensor<32xi32>
//      CHECK:   return %[[RESULT]]#0

// -----

func.func @scan_2d_memref(%0: memref<16x32xi32>, %1: memref<16x32xi32>) {
  %c0 = memref.alloc() : memref<32xi32>
  iree_linalg_ext.scan
    {__internal_linalg_transform__ = "outer_reduce_input"}
    dimension(0) inclusive(true)
    ins(%0 : memref<16x32xi32>) outs(%1, %c0 : memref<16x32xi32>, memref<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
//      CHECK:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//      CHECK:  func.func @scan_2d_memref(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
//      CHECK:    %[[C0:.+]] = arith.constant 0 : index
//      CHECK:    %[[C16:.+]] = arith.constant 16 : index
//      CHECK:    %[[C32:.+]] = arith.constant 32 : index
//      CHECK:    %[[C20:.+]] = arith.constant 20 : index
//      CHECK:    %[[ACC:.+]] = memref.alloc() : memref<32xi32>
//      CHECK:    scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C20]]
//      CHECK:      %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C20]], %[[C32]]]
//      CHECK:      %[[UPDATE_SLICE_IN:.+]] = memref.subview %[[ARG0]][0, %[[I]]] [%[[C16]], %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_OUT:.+]] = memref.subview %[[ARG1]][0, %[[I]]] [%[[C16]], %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_ACC:.+]] = memref.subview %[[ACC]][%[[I]]] [%[[SIZE]]]
//      CHECK:      iree_linalg_ext.scan
// CHECK-SAME:       {__internal_linalg_transform__ = "outer_reduce_output"}
// CHECK-SAME:       dimension(0) inclusive(true)
// CHECK-SAME:       ins(%[[UPDATE_SLICE_IN]]
// CHECK-SAME:       outs(%[[UPDATE_SLICE_OUT]], %[[UPDATE_SLICE_ACC]]
//      CHECK:   return

// -----

func.func @topk_tile_tensor(%input_values: tensor<?x?xf32>, %input_indices: tensor<?x?xi32>, %out_values: tensor<?x3xf32> , %out_indices: tensor<?x3xi32>) -> (tensor<?x3xf32>, tensor<?x3xi32>) {
  %0:2 = iree_linalg_ext.topk
        {__internal_linalg_transform__ = "inner_reduce_input"}
        dimension(1)
        ins(%input_values, %input_indices : tensor<?x?xf32> , tensor<?x?xi32>)
        outs(%out_values, %out_indices : tensor<?x3xf32>, tensor<?x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<?x3xf32>, tensor<?x3xi32>
  return %0#0, %0#1 : tensor<?x3xf32>, tensor<?x3xi32>
}

// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-LABEL: func.func @topk_tile_tensor
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[D0:.+]] = tensor.dim %[[ARG0:.+]], %[[C0]]
// CHECK:         %[[D1:.+]] = tensor.dim %[[ARG0:.+]], %[[C1]]
// CHECK:         %[[RESULT:.+]]:2 = scf.for %[[ARG4:.+]] = %[[C0]] to %[[D0]] step %[[C10]] iter_args(%[[ARG5:.+]] = %[[ARG2]], %[[ARG6:.+]] = %[[ARG3]])
// CHECK:           %[[D3:.+]] = affine.min #[[MAP0]](%[[ARG4]])[%[[C10]], %[[D0]]]
// CHECK:           %[[D4:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG4]], 0] [%[[D3]], %[[D1]]] [1, 1]
// CHECK:           %[[D5:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0] [%[[D3]], %[[D1]]] [1, 1]
// CHECK:           %[[D6:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG4]], 0] [%[[D3]], 3] [1, 1]
// CHECK:           %[[D7:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG4]], 0] [%[[D3]], 3] [1, 1]
// CHECK:           %[[D8:.+]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "inner_reduce_output"}
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[D4]], %[[D5]]
// CHECK-SAME:      outs(%[[D6]], %[[D7]]
// CHECK:           %[[D9:.+]] = tensor.insert_slice %[[D8]]#0 into %[[ARG5]][%[[ARG4]], 0] [%[[D3]], 3] [1, 1]
// CHECK:           %[[D10:.+]] = tensor.insert_slice %[[D8]]#1 into %[[ARG6]][%[[ARG4]], 0] [%[[D3]], 3] [1, 1]
// CHECK:           scf.yield %[[D9]], %[[D10]]
// CHECK:           return %[[RESULT]]#0, %[[RESULT]]#1


// -----

func.func @topk_tile_memref(%input_values: memref<?x?xf32>, %input_indices: memref<?x?xi32>, %out_values: memref<?x3xf32>, %out_indices: memref<?x3xi32>) {
  iree_linalg_ext.topk
        {__internal_linalg_transform__ = "inner_reduce_input"}
        dimension(1)
        ins(%input_values, %input_indices : memref<?x?xf32> , memref<?x?xi32>)
        outs(%out_values, %out_indices : memref<?x3xf32>, memref<?x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        }
  return
}

// CHECK:       #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-LABEL: func.func @topk_tile_memref
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[D0:.+]] = memref.dim %[[ARG0:.+]], %[[C0]]
// CHECK:         %[[D1:.+]] = memref.dim %[[ARG0:.+]], %[[C1]]
// CHECK:         scf.for %[[ARG4:.+]] = %[[C0]] to %[[D0]] step %[[C10]]
// CHECK:           %[[D2:.+]] = affine.min #[[MAP0]](%[[ARG4]])[%[[C10]], %[[D0]]]
// CHECK:           %[[D3:.+]] = memref.subview %[[ARG0]][%[[ARG4]], 0] [%[[D2]], %[[D1]]] [1, 1]
// CHECK:           %[[D4:.+]] = memref.subview %[[ARG1]][%[[ARG4]], 0] [%[[D2]], %[[D1]]] [1, 1]
// CHECK:           %[[D5:.+]] = memref.subview %[[ARG2]][%[[ARG4]], 0] [%[[D2]], 3] [1, 1]
// CHECK:           %[[D6:.+]] = memref.subview %[[ARG3]][%[[ARG4]], 0] [%[[D2]], 3] [1, 1]
// CHECK:           iree_linalg_ext.topk {__internal_linalg_transform__ = "inner_reduce_output"}
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[D3]], %[[D4]]
// CHECK-SAME:      outs(%[[D5]], %[[D6]]
// CHECK:           return

// -----

func.func @topk_tile_tensor_optional(%input_values: tensor<20x10xf32>, %out_values: tensor<20x3xf32> , %out_indices: tensor<20x3xi32>) -> (tensor<20x3xf32>, tensor<20x3xi32>) {
  %0:2 = iree_linalg_ext.topk
        {__internal_linalg_transform__ = "inner_reduce_input"}
        dimension(1)
        ins(%input_values : tensor<20x10xf32>)
        outs(%out_values, %out_indices : tensor<20x3xf32>, tensor<20x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<20x3xf32>, tensor<20x3xi32>
  return %0#0, %0#1 : tensor<20x3xf32>, tensor<20x3xi32>
}

// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-LABEL: func.func @topk_tile_tensor_optional
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C20:.+]] = arith.constant 20 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[RESULT:.+]]:2 = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C20]] step %[[C10]] iter_args(%[[ARG4:.+]] = %[[ARG1]], %[[ARG5:.+]] = %[[ARG2]])
// CHECK:           %[[D1:.+]] = affine.min #[[MAP0]](%[[ARG3]])[%[[C10]], %[[C20]]]
// CHECK:           %[[D2:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0] [%[[D1]], %[[C10]]] [1, 1]
// CHECK:           %[[D3:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0] [%[[D1]], 3] [1, 1]
// CHECK:           %[[D4:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], 0] [%[[D1]], 3] [1, 1]
// CHECK:           %[[D5:.+]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "inner_reduce_output"}
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[D2]]
// CHECK-SAME:      outs(%[[D3]], %[[D4]]
// CHECK:           %[[D6:.+]] = tensor.insert_slice %[[D5]]#0 into %[[ARG4]][%[[ARG3]], 0] [%[[D1]], 3] [1, 1]
// CHECK:           %[[D7:.+]] = tensor.insert_slice %[[D5]]#1 into %[[ARG5]][%[[ARG3]], 0] [%[[D1]], 3] [1, 1]
// CHECK:           scf.yield %[[D6]], %[[D7]]
// CHECK:           return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @winograd_input_transform(%arg0: tensor<1x10x10x1280xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform {__internal_linalg_transform__ = "tiling_winograd_input_nhwc"}
    output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<1x10x10x1280xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C1280:.+]] = arith.constant 1280 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:        %[[D2:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D3:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:          %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D2]], 10,
// CHECK-SAME:         10, %[[D4]]] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[D0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D2]], 2, 2, %[[D4]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// CHECK-SAME:         tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:         image_dimensions([1, 2]) ins(%[[EXTRACTED_SLICE]] : tensor<?x10x10x?xf32>)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE]]_0 : tensor<8x8x?x2x2x?xf32>) -> tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// CHECK-SAME:         %[[ARG3]]] [8, 8, %[[D2]], 2, 2, %[[D4]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// CHECK-SAME:         tensor<8x8x1x2x2x1280xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D3]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }

// -----

func.func @winograd_input_transform_memref(%arg0: memref<1x10x10x1280xf32>, %arg1: memref<8x8x1x2x2x1280xf32>) {
  iree_linalg_ext.winograd.input_transform {__internal_linalg_transform__ = "tiling_winograd_input_nhwc"}
    output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : memref<1x10x10x1280xf32>) outs(%arg1 : memref<8x8x1x2x2x1280xf32>)
  return
}
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @winograd_input_transform_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<1x10x10x1280xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C1280:.+]] = arith.constant 1280 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-DAG:        %[[D0:.+]] = affine.min #[[MAP2]](%[[ARG2]])[%[[C1]], %[[C1]]]
// CHECK:          scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]] {
// CHECK-DAG:          %[[D1:.+]] = affine.min #[[MAP3]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// CHECK:            %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[ARG2]], 0, 0, %[[ARG3]]] [%[[D0]], 10, 10, %[[D1]]]
// CHECK-SAME:         [1, 1, 1, 1] : memref<1x10x10x1280xf32> to memref<?x10x10x?xf32, strided<[128000, 12800, 1280,
// CHECK-SAME:         1], offset: ?>>
// CHECK:            %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]][0, 0, %[[ARG2]], 0, 0, %[[ARG3]]] [8, 8, %[[D0]], 2,
// CHECK-SAME:         2, %[[D1]]] [1, 1, 1, 1, 1, 1] : memref<8x8x1x2x2x1280xf32> to memref<8x8x?x2x2x?xf32,
// CHECK-SAME:         strided<[40960, 5120, 5120, 2560, 1280, 1], offset: ?>>
// CHECK:            iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1,
// CHECK-SAME:         2]) ins(%[[SUBVIEW]] : memref<?x10x10x?xf32, strided<[128000, 12800, 1280, 1], offset: ?>>)
// CHECK-SAME:         outs(%[[SUBVIEW]]_0 : memref<8x8x?x2x2x?xf32, strided<[40960, 5120, 5120, 2560, 1280, 1], offset:
// CHECK-SAME:         ?>>)
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }

// -----

func.func @winograd_output_transform(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x12x12x32xf32> {
  %0 = tensor.empty() : tensor<1x12x12x32xf32>
  %1 = iree_linalg_ext.winograd.output_transform {__internal_linalg_transform__ = "tiling_winograd_input_nhwc"}
        output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
        ins(%arg0 : tensor<8x8x1x2x2x32xf32>) outs(%0 : tensor<1x12x12x32xf32>) -> tensor<1x12x12x32xf32>
  return %1 : tensor<1x12x12x32xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// CHECK-SAME:   tensor<1x12x12x32xf32> {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<1x12x12x32xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<1x12x12x32xf32>) {
// CHECK-DAG:        %[[D2:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D3:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x12x12x32xf32>) {
// CHECK-DAG:          %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D2]], 2, 2, %[[D4]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[D0]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D2]], 12,
// CHECK-SAME:         12, %[[D4]]] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<?x12x12x?xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:         image_dimensions([1, 2]) ins(%[[EXTRACTED_SLICE]] : tensor<8x8x?x2x2x?xf32>)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE]]_0 : tensor<?x12x12x?xf32>) -> tensor<?x12x12x?xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG4]][%[[ARG1]], 0, 0, %[[ARG3]]]
// CHECK-SAME:         [%[[D2]], 12, 12, %[[D4]]] [1, 1, 1, 1] : tensor<?x12x12x?xf32> into tensor<1x12x12x32xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<1x12x12x32xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D3]] : tensor<1x12x12x32xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<1x12x12x32xf32>
// CHECK:      }

// -----

func.func @winograd_output_transform_memref(%arg0: memref<8x8x1x2x2x32xf32>, %arg1: memref<1x12x12x32xf32>) {
  iree_linalg_ext.winograd.output_transform {__internal_linalg_transform__ = "tiling_winograd_input_nhwc"}
   output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
   ins(%arg0 : memref<8x8x1x2x2x32xf32>) outs(%arg1 : memref<1x12x12x32xf32>)
  return
}
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @winograd_output_transform_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<8x8x1x2x2x32xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<1x12x12x32xf32>) {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-DAG:        %[[D0:.+]] = affine.min #[[MAP2]](%[[ARG2]])[%[[C1]], %[[C1]]]
// CHECK:          scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C32]] {
// CHECK-DAG:          %[[D1:.+]] = affine.min #[[MAP3]](%[[ARG3]])[%[[C32]], %[[C32]]]
// CHECK:            %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, 0, %[[ARG2]], 0, 0, %[[ARG3]]] [8, 8, %[[D0]], 2, 2,
// CHECK-SAME:         %[[D1]]] [1, 1, 1, 1, 1, 1] : memref<8x8x1x2x2x32xf32> to memref<8x8x?x2x2x?xf32, strided<[1024,
// CHECK-SAME:         128, 128, 64, 32, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]][%[[ARG2]], 0, 0, %[[ARG3]]] [%[[D0]], 12, 12, %[[D1]]]
// CHECK-SAME:         [1, 1, 1, 1] : memref<1x12x12x32xf32> to memref<?x12x12x?xf32, strided<[4608, 384, 32, 1],
// CHECK-SAME:         offset: ?>>
// CHECK:            iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1,
// CHECK-SAME:         2]) ins(%[[SUBVIEW]] : memref<8x8x?x2x2x?xf32, strided<[1024, 128, 128, 64, 32, 1], offset: ?>>)
// CHECK-SAME:         outs(%[[SUBVIEW]]_0 : memref<?x12x12x?xf32, strided<[4608, 384, 32, 1], offset: ?>>)
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }

// -----

func.func @winograd_input_transform_nchw(%arg0: tensor<1x1280x10x10xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform {__internal_linalg_transform__ = "tiling_winograd_input_nhwc"}
    output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
    ins(%arg0 : tensor<1x1280x10x10xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @winograd_input_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1280x10x10xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C1280:.+]] = arith.constant 1280 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:        %[[D2:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D3:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:          %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 10, 10] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<?x?x10x10xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[D0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D2]], 2, 2, %[[D4]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// CHECK-SAME:         tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:         image_dimensions([2, 3]) ins(%[[EXTRACTED_SLICE]] : tensor<?x?x10x10xf32>)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE]]_0 : tensor<8x8x?x2x2x?xf32>) -> tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// CHECK-SAME:         %[[ARG3]]] [8, 8, %[[D2]], 2, 2, %[[D4]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// CHECK-SAME:         tensor<8x8x1x2x2x1280xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D3]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }
// CHECK:    }

// -----

func.func @winograd_output_transform_nchw(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32> {
  %0 = tensor.empty() : tensor<1x32x12x12xf32>
  %1 = iree_linalg_ext.winograd.output_transform {__internal_linalg_transform__ = "tiling_winograd_input_nhwc"}
        output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
        ins(%arg0 : tensor<8x8x1x2x2x32xf32>) outs(%0 : tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
  return %1 : tensor<1x32x12x12xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func.func @winograd_output_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// CHECK-SAME:   tensor<1x32x12x12xf32> {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<1x32x12x12xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<1x32x12x12xf32>) {
// CHECK-DAG:        %[[D2:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D3:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x32x12x12xf32>) {
// CHECK-DAG:          %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D2]], 2, 2, %[[D4]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[D0]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 12, 12] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<?x?x12x12xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:         image_dimensions([2, 3]) ins(%[[EXTRACTED_SLICE]] : tensor<8x8x?x2x2x?xf32>)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE]]_0 : tensor<?x?x12x12xf32>) -> tensor<?x?x12x12xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG4]][%[[ARG1]], %[[ARG3]], 0, 0]
// CHECK-SAME:         [%[[D2]], %[[D4]], 12, 12] [1, 1, 1, 1] : tensor<?x?x12x12xf32> into tensor<1x32x12x12xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<1x32x12x12xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D3]] : tensor<1x32x12x12xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<1x32x12x12xf32>
// CHECK:      }
// CHECK:    }

// -----

func.func @softmax(%arg0: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
  %0 = tensor.empty() : tensor<16x64x256xf32>
  %1 = iree_linalg_ext.softmax {__internal_linalg_transform__ = "distribute_input"}
         dimension(1) ins(%arg0 : tensor<16x64x256xf32>) outs(%0 : tensor<16x64x256xf32>) -> tensor<16x64x256xf32>
  return %1 : tensor<16x64x256xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
// CHECK:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
// CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:        %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG:        %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:        %[[C30:.+]] = arith.constant 30 : index
// CHECK-DAG:        %[[D0:.+]] = tensor.empty() : tensor<16x64x256xf32>
// CHECK-DAG:        %[[D1:.+]] = iree_input.dispatch.workgroup.id[0] : index
// CHECK-DAG:        %[[D2:.+]] = iree_input.dispatch.workgroup.count[0] : index
// CHECK-DAG:        %[[D3:.+]] = iree_input.dispatch.workgroup.id[1] : index
// CHECK-DAG:        %[[D4:.+]] = iree_input.dispatch.workgroup.count[1] : index
// CHECK-DAG:      %[[D5:.+]] = affine.apply #[[MAP]]()[%[[D3]]]
// CHECK-DAG:      %[[D6:.+]] = affine.apply #[[MAP]]()[%[[D4]]]
// CHECK:        %[[D7:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[D5]] to %[[C16]] step %[[D6]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<16x64x256xf32>) {
// CHECK-DAG:        %[[D8:.+]] = affine.min #[[MAP1]](%[[ARG1]])[%[[C10]], %[[C16]]]
// CHECK-DAG:        %[[D9:.+]] = affine.apply #[[MAP2]]()[%[[D1]]]
// CHECK-DAG:        %[[D10:.+]] = affine.apply #[[MAP2]]()[%[[D2]]]
// CHECK:          %[[D11:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[D9]] to %[[C256]] step %[[D10]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<16x64x256xf32>) {
// CHECK-DAG:          %[[D12:.+]] = affine.min #[[MAP3]](%[[ARG3]])[%[[C30]], %[[C256]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, %[[ARG3]]] [%[[D8]],
// CHECK-SAME:         %[[C64]], %[[D12]]] [1, 1, 1] : tensor<16x64x256xf32> to tensor<?x?x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[D0]][%[[ARG1]], 0, %[[ARG3]]] [%[[D8]],
// CHECK-SAME:         %[[C64]], %[[D12]]] [1, 1, 1] : tensor<16x64x256xf32> to tensor<?x?x?xf32>
// CHECK:            %[[D13:.+]] = iree_linalg_ext.softmax {__internal_linalg_transform__ = "distribute_output"}
// CHECK-SAME:         dimension(1) ins(%[[EXTRACTED_SLICE]] : tensor<?x?x?xf32>) outs(%[[EXTRACTED_SLICE_0]] :
// CHECK-SAME:         tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D13]] into %[[ARG4]][%[[ARG1]], 0, %[[ARG3]]]
// CHECK-SAME:         [%[[D8]], %[[C64]], %[[D12]]] [1, 1, 1] : tensor<?x?x?xf32> into tensor<16x64x256xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<16x64x256xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D11]] : tensor<16x64x256xf32>
// CHECK:        }
// CHECK:        return %[[D7]] : tensor<16x64x256xf32>
// CHECK:      }

// -----

func.func @softmax_memref(%arg0: memref<16x64x256xf32>, %arg1: memref<16x64x256xf32>) {
  iree_linalg_ext.softmax {__internal_linalg_transform__ = "distribute_input"}
    dimension(1) ins(%arg0 : memref<16x64x256xf32>) outs(%arg1 : memref<16x64x256xf32>)
  return
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
// CHECK:      func.func @softmax_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<16x64x256xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   memref<16x64x256xf32>) {
// CHECK-DAG:    %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:    %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:    %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[C30:.+]] = arith.constant 30 : index
// CHECK:        %[[D0:.+]] = iree_input.dispatch.workgroup.id[0] : index
// CHECK:        %[[D1:.+]] = iree_input.dispatch.workgroup.count[0] : index
// CHECK:        %[[D2:.+]] = iree_input.dispatch.workgroup.id[1] : index
// CHECK:        %[[D3:.+]] = iree_input.dispatch.workgroup.count[1] : index
// CHECK-DAG:      %[[D4:.+]] = affine.apply #[[MAP]]()[%[[D2]]]
// CHECK-DAG:      %[[D5:.+]] = affine.apply #[[MAP]]()[%[[D3]]]
// CHECK:        scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[D4]] to %[[C16]] step %[[D5]] {
// CHECK-DAG:        %[[D6:.+]] = affine.min #[[MAP1]](%[[ARG2]])[%[[C10]], %[[C16]]]
// CHECK-DAG:        %[[D7:.+]] = affine.apply #[[MAP2]]()[%[[D0]]]
// CHECK-DAG:        %[[D8:.+]] = affine.apply #[[MAP2]]()[%[[D1]]]
// CHECK:          scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[D7]] to %[[C256]] step %[[D8]] {
// CHECK-DAG:          %[[D9:.+]] = affine.min #[[MAP3]](%[[ARG3]])[%[[C30]], %[[C256]]]
// CHECK:            %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[ARG2]], 0, %[[ARG3]]] [%[[D6]], %[[C64]], %[[D9]]]
// CHECK-SAME:         [1, 1, 1] : memref<16x64x256xf32> to memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]][%[[ARG2]], 0, %[[ARG3]]] [%[[D6]], %[[C64]], %[[D9]]]
// CHECK-SAME:         [1, 1, 1] : memref<16x64x256xf32> to memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>
// CHECK:            iree_linalg_ext.softmax {__internal_linalg_transform__ = "distribute_output"} dimension(1)
// CHECK-SAME:         ins(%[[SUBVIEW]] : memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>) outs(%[[SUBVIEW_0]] :
// CHECK-SAME:         memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>)
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }

// -----

func.func @attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = iree_linalg_ext.attention {__internal_linalg_transform__ = "tiling_attention"} ins(%query, %key, %value : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
// CHECK:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK:        %[[C30:.+]] = arith.constant 30 : index
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C192:.+]] = arith.constant 192 : index
// CHECK:        %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[C10:.+]] = arith.constant 10 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C10]]
// CHECK-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:        %[[D2:.+]] = affine.min #[[MAP]](%[[ARG3]])[%[[C10]], %[[C192]]]
// CHECK:          %[[D3:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C30]]
// CHECK-SAME:       iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:          %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG5]])[%[[C30]], %[[C1024]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[ARG5]], 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, 0] [%[[D2]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], 0, 0] [%[[D2]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D0]][%[[ARG3]], %[[ARG5]], 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.attention ins(%[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_0]],
// CHECK-SAME:         %[[EXTRACTED_SLICE_1]] : tensor<?x?x64xf32>, tensor<?x1024x64xf32>, tensor<?x1024x64xf32>)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE_2]] : tensor<?x?x64xf32>) -> tensor<?x?x64xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0]
// CHECK-SAME:         [%[[D2]], %[[D4]], 64] [1, 1, 1] : tensor<?x?x64xf32> into tensor<192x1024x64xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<192x1024x64xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D3]] : tensor<192x1024x64xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }

// -----

func.func @attention_memref(%query: memref<192x1024x64xf32>, %key: memref<192x1024x64xf32>, %value: memref<192x1024x64xf32>, %output: memref<192x1024x64xf32>) {
  iree_linalg_ext.attention {__internal_linalg_transform__ = "tiling_attention"} ins(%query, %key, %value : memref<192x1024x64xf32>, memref<192x1024x64xf32>, memref<192x1024x64xf32>) outs(%output : memref<192x1024x64xf32>)
  return
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
// CHECK:      func.func @attention_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   memref<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: memref<192x1024x64xf32>, %[[ARG3:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   memref<192x1024x64xf32>) {
// CHECK:        %[[C30:.+]] = arith.constant 30 : index
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C192:.+]] = arith.constant 192 : index
// CHECK:        %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[C10:.+]] = arith.constant 10 : index
// CHECK:        scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C10]] {
// CHECK-DAG:        %[[D0:.+]] = affine.min #[[MAP]](%[[ARG4]])[%[[C10]], %[[C192]]]
// CHECK:          scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C30]] {
// CHECK-DAG:          %[[D1:.+]] = affine.min #[[MAP1]](%[[ARG5]])[%[[C30]], %[[C1024]]]
// CHECK:            %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[ARG4]], %[[ARG5]], 0] [%[[D0]], %[[D1]], 64] [1, 1,
// CHECK-SAME:         1] : memref<192x1024x64xf32> to memref<?x?x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]][%[[ARG4]], 0, 0] [%[[D0]], 1024, 64] [1, 1, 1] :
// CHECK-SAME:         memref<192x1024x64xf32> to memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_1:.+]] = memref.subview %[[ARG2]][%[[ARG4]], 0, 0] [%[[D0]], 1024, 64] [1, 1, 1] :
// CHECK-SAME:         memref<192x1024x64xf32> to memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_2:.+]] = memref.subview %[[ARG3]][%[[ARG4]], %[[ARG5]], 0] [%[[D0]], %[[D1]], 64] [1, 1,
// CHECK-SAME:         1] : memref<192x1024x64xf32> to memref<?x?x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            iree_linalg_ext.attention ins(%[[SUBVIEW]], %[[SUBVIEW_0]], %[[SUBVIEW_1]] : memref<?x?x64xf32,
// CHECK-SAME:         strided<[65536, 64, 1], offset: ?>>, memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>,
// CHECK-SAME:         memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>) outs(%[[SUBVIEW_2]] :
// CHECK-SAME:         memref<?x?x64xf32, strided<[65536, 64, 1], offset: ?>>)
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }
