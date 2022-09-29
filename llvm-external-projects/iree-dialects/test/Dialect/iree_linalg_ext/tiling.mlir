// RUN: iree-dialects-opt --iree-linalg-ext-tile --split-input-file --verify-diagnostics -cse %s | FileCheck  %s

func.func @scatter_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
    {__internal_linalg_transform__ = "tiling_input"}
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
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xi32>
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
// CHECK:        %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]] : tensor<?x?xi32>
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
  %c0 = linalg.init_tensor [] : tensor<i32>
  %1 = linalg.init_tensor [128] : tensor<128xi32>
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
//      CHECK:   %[[ACC:.+]] = linalg.init_tensor [] : tensor<i32>
//      CHECK:   %[[OUTPUT:.+]] = linalg.init_tensor [128] : tensor<128xi32>
//      CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.scan
// CHECK-SAME:           __internal_linalg_transform__ = "outer_reduce_output"
// CHECK-SAME:       ins(%[[OPERAND]] :
// CHECK-SAME:       outs(%[[OUTPUT]], %[[ACC]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @scan_2d(%0: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %c0 = linalg.init_tensor [32] : tensor<32xi32>
  %1 = linalg.init_tensor [16, 32] : tensor<16x32xi32>
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
//      CHECK:    %[[ACC:.+]] = linalg.init_tensor [32] : tensor<32xi32>
//      CHECK:    %[[OUTPUT:.+]] = linalg.init_tensor [16, 32] : tensor<16x32xi32>
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
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[D0:.+]] = tensor.dim %[[ARG0:.+]], %[[C0]]
// CHECK:         %[[D1:.+]] = tensor.dim %[[ARG0:.+]], %[[C1]]
// CHECK:         %[[RESULT:.+]]:2 = scf.for %[[ARG4:.+]] = %[[C0]] to %[[D0]] step %[[C10]] iter_args(%[[ARG5:.+]] = %[[ARG2]], %[[ARG6:.+]] = %[[ARG3]])
// CHECK:           %[[D3:.+]] = affine.min #[[MAP0]](%[[ARG4]])[%[[C10]], %[[D0]]]
// CHECK:           %[[D4:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG4]], 0] [%[[D3]], %[[D1]]] [1, 1]
// CHECK:           %[[D5:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0] [%[[D3]], %[[D1]]] [1, 1]
// CHECK:           %[[D6:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG4]], 0] [%[[D3]], %[[C3]]] [1, 1]
// CHECK:           %[[D7:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG4]], 0] [%[[D3]], %[[C3]]] [1, 1]
// CHECK:           %[[D8:.+]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "inner_reduce_output"}
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[D4]], %[[D5]]
// CHECK-SAME:      outs(%[[D6]], %[[D7]]
// CHECK:           %[[D9:.+]] = tensor.insert_slice %[[D8]]#0 into %[[ARG5]][%[[ARG4]], 0] [%[[D3]], %[[C3]]] [1, 1]
// CHECK:           %[[D10:.+]] = tensor.insert_slice %[[D8]]#1 into %[[ARG6]][%[[ARG4]], 0] [%[[D3]], %[[C3]]] [1, 1]
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
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[D0:.+]] = memref.dim %[[ARG0:.+]], %[[C0]]
// CHECK:         %[[D1:.+]] = memref.dim %[[ARG0:.+]], %[[C1]]
// CHECK:         scf.for %[[ARG4:.+]] = %[[C0]] to %[[D0]] step %[[C10]]
// CHECK:           %[[D2:.+]] = affine.min #[[MAP0]](%[[ARG4]])[%[[C10]], %[[D0]]]
// CHECK:           %[[D3:.+]] = memref.subview %[[ARG0]][%[[ARG4]], 0] [%[[D2]], %[[D1]]] [1, 1]
// CHECK:           %[[D4:.+]] = memref.subview %[[ARG1]][%[[ARG4]], 0] [%[[D2]], %[[D1]]] [1, 1]
// CHECK:           %[[D5:.+]] = memref.subview %[[ARG2]][%[[ARG4]], 0] [%[[D2]], %[[C3]]] [1, 1]
// CHECK:           %[[D6:.+]] = memref.subview %[[ARG3]][%[[ARG4]], 0] [%[[D2]], %[[C3]]] [1, 1]
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
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[RESULT:.+]]:2 = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C20]] step %[[C10]] iter_args(%[[ARG4:.+]] = %[[ARG1]], %[[ARG5:.+]] = %[[ARG2]])
// CHECK:           %[[D1:.+]] = affine.min #[[MAP0]](%[[ARG3]])[%[[C10]], %[[C20]]]
// CHECK:           %[[D2:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0] [%[[D1]], %[[C10]]] [1, 1]
// CHECK:           %[[D3:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0] [%[[D1]], %[[C3]]] [1, 1]
// CHECK:           %[[D4:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], 0] [%[[D1]], %[[C3]]] [1, 1]
// CHECK:           %[[D5:.+]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "inner_reduce_output"}
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[D2]]
// CHECK-SAME:      outs(%[[D3]], %[[D4]]
// CHECK:           %[[D6:.+]] = tensor.insert_slice %[[D5]]#0 into %[[ARG4]][%[[ARG3]], 0] [%[[D1]], %[[C3]]] [1, 1]
// CHECK:           %[[D7:.+]] = tensor.insert_slice %[[D5]]#1 into %[[ARG5]][%[[ARG3]], 0] [%[[D1]], %[[C3]]] [1, 1]
// CHECK:           scf.yield %[[D6]], %[[D7]]
// CHECK:           return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @NC_to_NCnc(%arg0: tensor<128x256xf32>, %arg1: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %0 = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_input"} %arg0 dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : (tensor<128x256xf32> tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32>
  return %0 : tensor<4x8x32x32xf32>
}
// CHECK:         #map0 = affine_map<(d0)[s0, s1] -> (2, -d0 + s1)>
// CHECK:         #map1 = affine_map<(d0)[s0, s1] -> (4, -d0 + s1)>
// CHECK:         #map2 = affine_map<(d0) -> (d0 * 32)>
// CHECK:         #map3 = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK:         #map4 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @NC_to_NCnc(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<128x256xf32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 256 : index
// CHECK:           %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_10:.*]] = %[[VAL_1]]) -> (tensor<4x8x32x32xf32>) {
// CHECK:             %[[VAL_11:.*]] = affine.min #map0(%[[VAL_9]]){{\[}}%[[VAL_5]], %[[VAL_3]]]
// CHECK:             %[[VAL_12:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_14:.*]] = %[[VAL_10]]) -> (tensor<4x8x32x32xf32>) {
// CHECK:               %[[VAL_15:.*]] = affine.min #map1(%[[VAL_13]]){{\[}}%[[VAL_3]], %[[VAL_4]]]
// CHECK:               %[[VAL_16:.*]] = affine.apply #map2(%[[VAL_9]])
// CHECK:               %[[VAL_17:.*]] = affine.apply #map2(%[[VAL_11]])
// CHECK:               %[[VAL_18:.*]] = affine.apply #map3(%[[VAL_6]], %[[VAL_16]])
// CHECK:               %[[VAL_19:.*]] = affine.min #map4(%[[VAL_17]], %[[VAL_18]])
// CHECK:               %[[VAL_20:.*]] = affine.apply #map2(%[[VAL_13]])
// CHECK:               %[[VAL_21:.*]] = affine.apply #map2(%[[VAL_15]])
// CHECK:               %[[VAL_22:.*]] = affine.apply #map3(%[[VAL_7]], %[[VAL_20]])
// CHECK:               %[[VAL_23:.*]] = affine.min #map4(%[[VAL_21]], %[[VAL_22]])
// CHECK:               %[[VAL_24:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_16]], %[[VAL_20]]] {{\[}}%[[VAL_19]], %[[VAL_23]]] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
// CHECK:               %[[VAL_25:.*]] = tensor.extract_slice %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_13]], 0, 0] {{\[}}%[[VAL_11]], %[[VAL_15]], 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<?x?x32x32xf32>
// CHECK:               %[[VAL_26:.*]] = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_output"} %[[VAL_24]] dims_pos = [0, 1] inner_tiles = [32, 32] into %[[VAL_25]] : (tensor<?x?xf32> tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
// CHECK:               %[[VAL_27:.*]] = tensor.insert_slice %[[VAL_26]] into %[[VAL_14]]{{\[}}%[[VAL_9]], %[[VAL_13]], 0, 0] {{\[}}%[[VAL_11]], %[[VAL_15]], 32, 32] [1, 1, 1, 1] : tensor<?x?x32x32xf32> into tensor<4x8x32x32xf32>
// CHECK:               scf.yield %[[VAL_27]] : tensor<4x8x32x32xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_28:.*]] : tensor<4x8x32x32xf32>
// CHECK:           }
// CHECK:           return %[[VAL_29:.*]] : tensor<4x8x32x32xf32>
// CHECK:         }

// -----

func.func @pad_and_pack_static(%input: tensor<13x15xf32>, %output: tensor<2x8x8x2xf32>, %pad: f32) -> tensor<2x8x8x2xf32> {
  %0 = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_input"} %input padding_value(%pad : f32) dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<13x15xf32> tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK:         #map0 = affine_map<(d0)[s0, s1] -> (2, -d0 + s1)>
// CHECK:         #map1 = affine_map<(d0)[s0, s1] -> (4, -d0 + s1)>
// CHECK:         #map2 = affine_map<(d0) -> (d0 * 8)>
// CHECK:         #map3 = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK:         #map4 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:         #map5 = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:   func.func @pad_and_pack_static(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<13x15xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<2x8x8x2xf32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: f32) -> tensor<2x8x8x2xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 13 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 15 : index
// CHECK:           %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_4]] iter_args(%[[VAL_11:.*]] = %[[VAL_1]]) -> (tensor<2x8x8x2xf32>) {
// CHECK:             %[[VAL_12:.*]] = affine.min #map0(%[[VAL_10]]){{\[}}%[[VAL_4]], %[[VAL_4]]]
// CHECK:             %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_15:.*]] = %[[VAL_11]]) -> (tensor<2x8x8x2xf32>) {
// CHECK:               %[[VAL_16:.*]] = affine.min #map1(%[[VAL_14]]){{\[}}%[[VAL_6]], %[[VAL_5]]]
// CHECK:               %[[VAL_17:.*]] = affine.apply #map2(%[[VAL_10]])
// CHECK:               %[[VAL_18:.*]] = affine.apply #map2(%[[VAL_12]])
// CHECK:               %[[VAL_19:.*]] = affine.apply #map3(%[[VAL_7]], %[[VAL_17]])
// CHECK:               %[[VAL_20:.*]] = affine.min #map4(%[[VAL_18]], %[[VAL_19]])
// CHECK:               %[[VAL_21:.*]] = affine.apply #map5(%[[VAL_14]])
// CHECK:               %[[VAL_22:.*]] = affine.apply #map5(%[[VAL_16]])
// CHECK:               %[[VAL_23:.*]] = affine.apply #map3(%[[VAL_8]], %[[VAL_21]])
// CHECK:               %[[VAL_24:.*]] = affine.min #map4(%[[VAL_22]], %[[VAL_23]])
// CHECK:               %[[VAL_25:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_21]]] {{\[}}%[[VAL_20]], %[[VAL_24]]] [1, 1] : tensor<13x15xf32> to tensor<?x?xf32>
// CHECK:               %[[VAL_26:.*]] = tensor.extract_slice %[[VAL_1]]{{\[}}%[[VAL_10]], %[[VAL_14]], 0, 0] {{\[}}%[[VAL_12]], %[[VAL_16]], 8, 2] [1, 1, 1, 1] : tensor<2x8x8x2xf32> to tensor<?x?x8x2xf32>
// CHECK:               %[[VAL_27:.*]] = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_output"} %[[VAL_25]] padding_value(%[[VAL_2]] : f32) dims_pos = [0, 1] inner_tiles = [8, 2] into %[[VAL_26]] : (tensor<?x?xf32> tensor<?x?x8x2xf32>) -> tensor<?x?x8x2xf32>
// CHECK:               %[[VAL_28:.*]] = tensor.insert_slice %[[VAL_27]] into %[[VAL_15]]{{\[}}%[[VAL_10]], %[[VAL_14]], 0, 0] {{\[}}%[[VAL_12]], %[[VAL_16]], 8, 2] [1, 1, 1, 1] : tensor<?x?x8x2xf32> into tensor<2x8x8x2xf32>
// CHECK:               scf.yield %[[VAL_28]] : tensor<2x8x8x2xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_29:.*]] : tensor<2x8x8x2xf32>
// CHECK:           }
// CHECK:           return %[[VAL_30:.*]] : tensor<2x8x8x2xf32>
// CHECK:         }

// -----

func.func @pad_and_pack_partially_dynamic(%input: tensor<?x?xf32>, %output: tensor<?x?x8x2xf32>, %pad: f32) -> tensor<?x?x8x2xf32> {
  %0 = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_input"} %input padding_value(%pad : f32) dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<?x?xf32> tensor<?x?x8x2xf32>) -> tensor<?x?x8x2xf32>
  return %0 : tensor<?x?x8x2xf32>
}
// CHECK:         #map0 = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK:         #map1 = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK:         #map2 = affine_map<(d0)[s0, s1] -> (2, -d0 + s1)>
// CHECK:         #map3 = affine_map<(d0)[s0, s1] -> (4, -d0 + s1)>
// CHECK:         #map4 = affine_map<(d0) -> (d0 * 8)>
// CHECK:         #map5 = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK:         #map6 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:         #map7 = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:   func.func @pad_and_pack_partially_dynamic(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: tensor<?x?x8x2xf32>,
// CHECK-SAME:                                              %[[VAL_2:.*]]: f32) -> tensor<?x?x8x2xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?x8x2xf32>
// CHECK:           %[[VAL_8:.*]] = affine.apply #map0(){{\[}}%[[VAL_7]]]
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?x8x2xf32>
// CHECK:           %[[VAL_10:.*]] = affine.apply #map1(){{\[}}%[[VAL_9]]]
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_3]] to %[[VAL_8]] step %[[VAL_5]] iter_args(%[[VAL_13:.*]] = %[[VAL_1]]) -> (tensor<?x?x8x2xf32>) {
// CHECK:             %[[VAL_14:.*]] = affine.min #map2(%[[VAL_12]]){{\[}}%[[VAL_5]], %[[VAL_8]]]
// CHECK:             %[[VAL_15:.*]] = scf.for %[[VAL_16:.*]] = %[[VAL_3]] to %[[VAL_10]] step %[[VAL_6]] iter_args(%[[VAL_17:.*]] = %[[VAL_13]]) -> (tensor<?x?x8x2xf32>) {
// CHECK:               %[[VAL_18:.*]] = affine.min #map3(%[[VAL_16]]){{\[}}%[[VAL_6]], %[[VAL_10]]]
// CHECK:               %[[VAL_19:.*]] = affine.apply #map4(%[[VAL_12]])
// CHECK:               %[[VAL_20:.*]] = affine.apply #map4(%[[VAL_14]])
// CHECK:               %[[VAL_21:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf32>
// CHECK:               %[[VAL_22:.*]] = affine.apply #map5(%[[VAL_21]], %[[VAL_19]])
// CHECK:               %[[VAL_23:.*]] = affine.min #map6(%[[VAL_20]], %[[VAL_22]])
// CHECK:               %[[VAL_24:.*]] = affine.apply #map7(%[[VAL_16]])
// CHECK:               %[[VAL_25:.*]] = affine.apply #map7(%[[VAL_18]])
// CHECK:               %[[VAL_26:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:               %[[VAL_27:.*]] = affine.apply #map5(%[[VAL_26]], %[[VAL_24]])
// CHECK:               %[[VAL_28:.*]] = affine.min #map6(%[[VAL_25]], %[[VAL_27]])
// CHECK:               %[[VAL_29:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_19]], %[[VAL_24]]] {{\[}}%[[VAL_23]], %[[VAL_28]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:               %[[VAL_30:.*]] = tensor.extract_slice %[[VAL_1]]{{\[}}%[[VAL_12]], %[[VAL_16]], 0, 0] {{\[}}%[[VAL_14]], %[[VAL_18]], 8, 2] [1, 1, 1, 1] : tensor<?x?x8x2xf32> to tensor<?x?x8x2xf32>
// CHECK:               %[[VAL_31:.*]] = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_output"} %[[VAL_29]] padding_value(%[[VAL_2]] : f32) dims_pos = [0, 1] inner_tiles = [8, 2] into %[[VAL_30]] : (tensor<?x?xf32> tensor<?x?x8x2xf32>) -> tensor<?x?x8x2xf32>
// CHECK:               %[[VAL_32:.*]] = tensor.insert_slice %[[VAL_31]] into %[[VAL_17]]{{\[}}%[[VAL_12]], %[[VAL_16]], 0, 0] {{\[}}%[[VAL_14]], %[[VAL_18]], 8, 2] [1, 1, 1, 1] : tensor<?x?x8x2xf32> into tensor<?x?x8x2xf32>
// CHECK:               scf.yield %[[VAL_32]] : tensor<?x?x8x2xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_33:.*]] : tensor<?x?x8x2xf32>
// CHECK:           }
// CHECK:           return %[[VAL_34:.*]] : tensor<?x?x8x2xf32>
// CHECK:         }

// -----

func.func @pad_and_pack_fully_dynamic(%input: tensor<?x?xf32>, %output: tensor<?x?x?x?xf32>, %pad: f32, %tile_n : index, %tile_m : index) -> tensor<?x?x?x?xf32> {
  %0 = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_input"} %input padding_value(%pad : f32)
    dims_pos = [0, 1] inner_tiles = [%tile_n, %tile_m] into %output : (tensor<?x?xf32> tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK:         #map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
// CHECK:         #map1 = affine_map<(d0)[s0, s1] -> (2, -d0 + s1)>
// CHECK:         #map2 = affine_map<(d0)[s0, s1] -> (4, -d0 + s1)>
// CHECK:         #map3 = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK:         #map4 = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK:         #map5 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @pad_and_pack_fully_dynamic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: f32,
// CHECK-SAME:                                          %[[VAL_3:.*]]: index,
// CHECK-SAME:                                          %[[VAL_4:.*]]: index) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_1]], %[[VAL_5]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_11:.*]] = affine.apply #map0(){{\[}}%[[VAL_10]], %[[VAL_3]]]
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_1]], %[[VAL_6]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_13:.*]] = affine.apply #map0(){{\[}}%[[VAL_12]], %[[VAL_4]]]
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_1]], %[[VAL_7]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_1]], %[[VAL_9]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_5]] to %[[VAL_11]] step %[[VAL_7]] iter_args(%[[VAL_18:.*]] = %[[VAL_1]]) -> (tensor<?x?x?x?xf32>) {
// CHECK:             %[[VAL_19:.*]] = affine.min #map1(%[[VAL_17]]){{\[}}%[[VAL_7]], %[[VAL_11]]]
// CHECK:             %[[VAL_20:.*]] = scf.for %[[VAL_21:.*]] = %[[VAL_5]] to %[[VAL_13]] step %[[VAL_8]] iter_args(%[[VAL_22:.*]] = %[[VAL_18]]) -> (tensor<?x?x?x?xf32>) {
// CHECK:               %[[VAL_23:.*]] = affine.min #map2(%[[VAL_21]]){{\[}}%[[VAL_8]], %[[VAL_13]]]
// CHECK:               %[[VAL_24:.*]] = affine.apply #map3(%[[VAL_17]]){{\[}}%[[VAL_3]]]
// CHECK:               %[[VAL_25:.*]] = affine.apply #map3(%[[VAL_19]]){{\[}}%[[VAL_3]]]
// CHECK:               %[[VAL_26:.*]] = tensor.dim %[[VAL_0]], %[[VAL_5]] : tensor<?x?xf32>
// CHECK:               %[[VAL_27:.*]] = affine.apply #map4(%[[VAL_26]], %[[VAL_24]])
// CHECK:               %[[VAL_28:.*]] = affine.min #map5(%[[VAL_25]], %[[VAL_27]])
// CHECK:               %[[VAL_29:.*]] = affine.apply #map3(%[[VAL_21]]){{\[}}%[[VAL_4]]]
// CHECK:               %[[VAL_30:.*]] = affine.apply #map3(%[[VAL_23]]){{\[}}%[[VAL_4]]]
// CHECK:               %[[VAL_31:.*]] = tensor.dim %[[VAL_0]], %[[VAL_6]] : tensor<?x?xf32>
// CHECK:               %[[VAL_32:.*]] = affine.apply #map4(%[[VAL_31]], %[[VAL_29]])
// CHECK:               %[[VAL_33:.*]] = affine.min #map5(%[[VAL_30]], %[[VAL_32]])
// CHECK:               %[[VAL_34:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_24]], %[[VAL_29]]] {{\[}}%[[VAL_28]], %[[VAL_33]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:               %[[VAL_35:.*]] = tensor.extract_slice %[[VAL_1]]{{\[}}%[[VAL_17]], %[[VAL_21]], 0, 0] {{\[}}%[[VAL_19]], %[[VAL_23]], %[[VAL_14]], %[[VAL_15]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
// CHECK:               %[[VAL_36:.*]] = iree_linalg_ext.pack {__internal_linalg_transform__ = "tiling_pack_output"} %[[VAL_34]] padding_value(%[[VAL_2]] : f32) dims_pos = [0, 1] inner_tiles = {{\[}}%[[VAL_3]], %[[VAL_4]]] into %[[VAL_35]] : (tensor<?x?xf32> tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:               %[[VAL_37:.*]] = tensor.insert_slice %[[VAL_36]] into %[[VAL_22]]{{\[}}%[[VAL_17]], %[[VAL_21]], 0, 0] {{\[}}%[[VAL_19]], %[[VAL_23]], %[[VAL_14]], %[[VAL_15]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> into tensor<?x?x?x?xf32>
// CHECK:               scf.yield %[[VAL_37]] : tensor<?x?x?x?xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_38:.*]] : tensor<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return %[[VAL_39:.*]] : tensor<?x?x?x?xf32>
// CHECK:         }

