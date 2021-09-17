// RUN: iree-opt -iree-linalg-ext-tile -split-input-file %s | IreeFileCheck %s

func @scatter_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg_ext.scatter
    {__internal_linalg_transform__ = "tiling_input"}
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func @scatter_tiling(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[TILESIZEY:.+]] = constant 10 : index
//   CHECK-DAG:   %[[TILESIZEX:.+]] = constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
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
//       CHECK:       %[[ORIGINAL_SLICE:.+]] = tensor.extract_slice %[[INITX]][0, %[[IV1]]]
//  CHECK-SAME:           [%[[SCATTER_DIM]], %[[USED_TILESIZEX]]]
//       CHECK:       %[[SCATTER_TILE:.+]] = linalg_ext.scatter
//  CHECK-SAME:           __internal_linalg_transform__ = "tiling_output"
//  CHECK-SAME:           ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:           outs(%[[ORIGINAL_SLICE]]
//       CHECK:       %[[YIELD:.+]] = tensor.insert_slice %[[SCATTER_TILE]] into %[[INITX]][0, %[[IV1]]]
//  CHECK-SAME:           [%[[SCATTER_DIM]], %[[USED_TILESIZEX]]]
//       CHECK:       scf.yield %[[YIELD]]
//       CHECK:     scf.yield %[[RESULT_INNER]]
//       CHECK:   return %[[RESULT]]

// -----

func @scatter_tiling_memref(
    %original: memref<?x?xf32>, %indices: memref<?x1xi32>,
    %update : memref<?x?xf32>) {
  linalg_ext.scatter
    {__internal_linalg_transform__ = "tiling_input"}
    ins(%update, %indices : memref<?x?xf32>, memref<?x1xi32>)
    outs(%original : memref<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    }
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func @scatter_tiling_memref(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[TILESIZEY:.+]] = constant 10 : index
//   CHECK-DAG:   %[[TILESIZEX:.+]] = constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
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
//       CHECK:       linalg_ext.scatter
//  CHECK-SAME:           __internal_linalg_transform__ = "tiling_output"
//  CHECK-SAME:           ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:           outs(%[[ORIGINAL_SLICE]]

// -----

func @scatter_tiling_distribution(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg_ext.scatter
    {__internal_linalg_transform__ = "distribute_input"}
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
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[UPDATES]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[UPDATES]], %[[C1]]
//   CHECK-DAG:   %[[ID:.+]] = flow.dispatch.workgroup.id[0]
//   CHECK-DAG:   %[[COUNT:.+]] = flow.dispatch.workgroup.count[0]
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
//       CHECK:     %[[ORIGINAL_SLICE:.+]] = tensor.extract_slice %[[INIT]][0, 0]
//  CHECK-SAME:         [%[[D2]], %[[D1]]]
//       CHECK:     %[[SCATTER_TILE:.+]] = linalg_ext.scatter
//  CHECK-SAME:        __internal_linalg_transform__ = "distribute_output"
//  CHECK-SAME:        ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:        outs(%[[ORIGINAL_SLICE]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SCATTER_TILE]] into %[[INIT]][0, 0]
//  CHECK-SAME:        [%[[D2]], %[[D1]]]
//       CHECK:   return %[[RESULT]]

// -----

func @scatter_no_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg_ext.scatter
    {__internal_linalg_transform__ = "no_tiling_input"}
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
//  CHECK-SAME:       __internal_linalg_transform__ = "no_tiling_output"
//  CHECK-SAME:       ins(%[[UPDATES]], %[[INDICES]]
//  CHECK-SAME:       outs(%[[ORIGINAL]]
//       CHECK:   return %[[RESULT]]

// -----

func @sort_1d(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = linalg_ext.sort
       {__internal_linalg_transform__ = "outer_reduce_input"}
       outs(%arg0 : tensor<?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
         %0 = cmpi sgt, %arg2, %arg3 : i32
         linalg_ext.yield %0 : i1
       } -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
//      CHECK: func @sort_1d(
// CHECK-SAME:   %[[OPERAND:.+]]: tensor<?xi32>
//      CHECK:   %[[RESULT:.+]] = linalg_ext.sort
// CHECK-SAME:       {__internal_linalg_transform__ = "outer_reduce_output"}
// CHECK-SAME:       outs(%[[OPERAND]] :
//      CHECK:   return %[[RESULT]]

// -----

func @sort_2d(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg_ext.sort dimension(1)
       {__internal_linalg_transform__ = "inner_reduce_input"}
       outs(%arg0 : tensor<?x?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
         %0 = cmpi sgt, %arg2, %arg3 : i32
         linalg_ext.yield %0 : i1
       } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func @sort_2d(
//  CHECK-SAME:   %[[OPERAND:.+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[OPERAND]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//       CHECK:     %[[OPERAND_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[SORT_TILE:.+]] = linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "inner_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND_SLICE]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SORT_TILE]] into %[[INIT]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// -----

func @sort_2d_inner_parallel(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg_ext.sort dimension(0)
       {__internal_linalg_transform__ = "outer_reduce_input"}
       outs(%arg0 : tensor<?x?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
         %0 = cmpi sgt, %arg2, %arg3 : i32
         linalg_ext.yield %0 : i1
       } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func @sort_2d_inner_parallel(
//  CHECK-SAME:   %[[OPERAND:.+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[OPERAND]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D1]]]
//       CHECK:     %[[OPERAND_SLICE:.+]] = tensor.extract_slice %[[INIT]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     %[[SORT_TILE:.+]] = linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "outer_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND_SLICE]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SORT_TILE]] into %[[INIT]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// -----

func @sort_2d_multi_result(
    %arg0: tensor<?x?xi32>, %arg1: tensor<?x?xf32>)
    -> (tensor<?x?xi32>, tensor<?x?xf32>) {
  %0:2 = linalg_ext.sort dimension(1)
       {__internal_linalg_transform__ = "inner_reduce_input"}
       outs(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xf32>) {
       ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
         %1 = cmpf ogt, %arg4, %arg5 : f32
         linalg_ext.yield %1 : i1
       } -> tensor<?x?xi32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xf32>
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//       CHECK: func @sort_2d_multi_result(
//  CHECK-SAME:   %[[OPERAND1:.+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:.+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND1]], %[[C1]]
//       CHECK:   %[[RESULT:.+]]:2 = scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT1:.+]] = %[[OPERAND1]], %[[INIT2:.+]] = %[[OPERAND2]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D0]]]
//       CHECK:     %[[OPERAND1_SLICE:.+]] = tensor.extract_slice %[[INIT1]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[OPERAND2_SLICE:.+]] = tensor.extract_slice %[[INIT2]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[SORT_TILE:.+]]:2 = linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "inner_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]
//       CHECK:     %[[YIELD1:.+]] = tensor.insert_slice %[[SORT_TILE]]#0 into %[[INIT1]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[YIELD2:.+]] = tensor.insert_slice %[[SORT_TILE]]#1 into %[[INIT2]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     scf.yield %[[YIELD1]], %[[YIELD2]]
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func @sort_2d_multi_result_memref(
    %arg0: memref<?x?xi32>, %arg1: memref<?x?xf32>) {
  linalg_ext.sort dimension(0)
     {__internal_linalg_transform__ = "outer_reduce_input"}
     outs(%arg0, %arg1 : memref<?x?xi32>, memref<?x?xf32>) {
     ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
       %0 = cmpf ogt, %arg4, %arg5 : f32
       linalg_ext.yield %0 : i1
     }
  return
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//       CHECK: func @sort_2d_multi_result_memref(
//  CHECK-SAME:   %[[OPERAND1:.+]]: memref<?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:.+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[OPERAND1]], %[[C1]]
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZE]]
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[TILESIZE]], %[[D1]]]
//       CHECK:     %[[OPERAND1_SLICE:.+]] = memref.subview %[[OPERAND1]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     %[[OPERAND2_SLICE:.+]] = memref.subview %[[OPERAND2]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     linalg_ext.sort
//  CHECK-SAME:         __internal_linalg_transform__ = "outer_reduce_output"
//  CHECK-SAME:         outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]

// -----

func @sort_3d_multi_result_distribute(
  %arg0: tensor<?x?x?xi32>, %arg1 : tensor<?x?x?xf32>)
  -> (tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
  %0, %1 = linalg_ext.sort dimension(1)
      {__internal_linalg_transform__ = "distribute_input"}
      outs(%arg0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %2 = cmpf ogt, %arg4, %arg5 : f32
        linalg_ext.yield %2 : i1
      } -> tensor<?x?x?xi32>, tensor<?x?x?xf32>
  return %0, %1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
//       CHECK: func @sort_3d_multi_result_distribute(
//  CHECK-SAME:   %[[OPERAND1:[a-zA-Z0-9_]+]]: tensor<?x?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//   CHECK-DAG:   %[[TILESIZE1:.+]] = constant 10 : index
//   CHECK-DAG:   %[[TILESIZE2:.+]] = constant 30 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND1]], %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[OPERAND1]], %[[C2]]
//   CHECK-DAG:   %[[IDX:.+]] = flow.dispatch.workgroup.id[0]
//   CHECK-DAG:   %[[COUNTX:.+]] = flow.dispatch.workgroup.count[0]
//   CHECK-DAG:   %[[IDY:.+]] = flow.dispatch.workgroup.id[1]
//   CHECK-DAG:   %[[COUNTY:.+]] = flow.dispatch.workgroup.count[1]
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
//       CHECK:       %[[OPERAND1_SLICE:.+]] = tensor.extract_slice %[[INIT3]][%[[IV0]], 0, %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZE1]], %[[D1]], %[[USED_TILESIZE2]]]
//       CHECK:       %[[OPERAND2_SLICE:.+]] = tensor.extract_slice %[[INIT4]][%[[IV0]], 0, %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZE1]], %[[D1]], %[[USED_TILESIZE2]]]
//       CHECK:       %[[SORT_SLICE:.+]]:2 = linalg_ext.sort
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

func @sort_3d_multi_result_distribute_memref(
  %arg0: memref<?x?x?xi32>, %arg1 : memref<?x?x?xf32>) {
  linalg_ext.sort dimension(1)
      {__internal_linalg_transform__ = "distribute_input"}
      outs(%arg0, %arg1 : memref<?x?x?xi32>, memref<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %0 = cmpf ogt, %arg4, %arg5 : f32
        linalg_ext.yield %0 : i1
      }
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
//       CHECK: func @sort_3d_multi_result_distribute_memref(
//  CHECK-SAME:   %[[OPERAND1:[a-zA-Z0-9_]+]]: memref<?x?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:[a-zA-Z0-9_]+]]: memref<?x?x?xf32>
//   CHECK-DAG:   %[[TILESIZE1:.+]] = constant 10 : index
//   CHECK-DAG:   %[[TILESIZE2:.+]] = constant 30 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[OPERAND1]], %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = memref.dim %[[OPERAND1]], %[[C2]]
//   CHECK-DAG:   %[[IDX:.+]] = flow.dispatch.workgroup.id[0]
//   CHECK-DAG:   %[[COUNTX:.+]] = flow.dispatch.workgroup.count[0]
//   CHECK-DAG:   %[[IDY:.+]] = flow.dispatch.workgroup.id[1]
//   CHECK-DAG:   %[[COUNTY:.+]] = flow.dispatch.workgroup.count[1]
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
//       CHECK:       linalg_ext.sort
//  CHECK-SAME:           __internal_linalg_transform__ = "distribute_output"
//  CHECK-SAME:           outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]

// -----

func @slice_insert(%source :tensor<?x?xf32>, %dest: tensor<?x?xf32>,
                   %idx0 : index, %idx1 : index) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = tensor.dim %source, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %source, %c1 : tensor<?x?xf32>
  %2 = tensor.insert_slice %source into %dest[%idx0, %idx1] [%0, %1] [1, 1]
      {__internal_linalg_transform__ = "tiling_input"} : tensor<?x?xf32> into tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//      CHECK: func @slice_insert(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] =
//      CHECK:     %[[YIELD1:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] =
//      CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[OFFSET0:.+]] = affine.apply #[[MAP2]](%[[IV0]])[%[[ARG2]]]
//      CHECK:       %[[OFFSET1:.+]] = affine.apply #[[MAP2]](%[[IV1]])[%[[ARG3]]]
//      CHECK:       %[[UPDATE:.+]] = tensor.insert_slice %[[SLICE]]
// CHECK-SAME:         into %{{.+}}[%[[OFFSET0]], %[[OFFSET1]]]
//      CHECK:       scf.yield %[[UPDATE]]
//      CHECK:     scf.yield %[[YIELD1]]
//      CHECK:   return %[[RESULT]]

// -----

func @slice_insert_rank_reduce(%source :tensor<?x?xf32>, %dest: tensor<?x?x?xf32>,
                   %idx0 : index, %idx1 : index) -> tensor<?x?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = tensor.dim %source, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %source, %c1 : tensor<?x?xf32>
  %2 = tensor.insert_slice %source into %dest[%idx0, 0, %idx1] [%0, 1, %1] [1, 1, 1]
      {__internal_linalg_transform__ = "tiling_input"} : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %2 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (20, -d0 + s1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//      CHECK: func @slice_insert_rank_reduce(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] =
//      CHECK:     %[[YIELD1:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] =
//      CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[IV1]]]
//      CHECK:       %[[OFFSET0:.+]] = affine.apply #[[MAP2]](%[[IV0]])[%[[ARG2]]]
//      CHECK:       %[[OFFSET1:.+]] = affine.apply #[[MAP2]](%[[IV1]])[%[[ARG3]]]
//      CHECK:       %[[UPDATE:.+]] = tensor.insert_slice %[[SLICE]]
// CHECK-SAME:         into %{{.+}}[%[[OFFSET0]], 0, %[[OFFSET1]]]
//      CHECK:       scf.yield %[[UPDATE]]
//      CHECK:     scf.yield %[[YIELD1]]
//      CHECK:   return %[[RESULT]]

// -----

func @fft_1d_stage_5(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>,
    %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  %cst1 = constant 5 : index
  %0:2 = linalg_ext.fft
  {__internal_linalg_transform__ = "tiling_1d_stage5_fft_input"}
    ins(%cst1, %arg2, %arg3: index, tensor<16xf32>, tensor<16xf32>)
    outs(%arg0, %arg1: tensor<1024xf32>, tensor<1024xf32>)
  : tensor<1024xf32>, tensor<1024xf32>
  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func @fft_1d_stage_5(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = constant 0 : index
// CHECK-DAG:    %[[C5:.+]] = constant 5 : index
// CHECK-DAG:    %[[C32:.+]] = constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = constant 1024 : index
// CHECK:        %[[RES:.+]]:2 = scf.for %[[I:.+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG5:.+]] = %[[ARG0]], %[[ARG6:.+]] = %[[ARG1]])
// CHECK-SAME:       -> (tensor<1024xf32>, tensor<1024xf32>) {
// CHECK:          %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C32]], %[[C1024]]]
// CHECK:          %[[SLICE1:.+]] = tensor.extract_slice %[[ARG5]][%[[I]]] [%[[SIZE]]] [1] : tensor<1024xf32> to tensor<?xf32>
// CHECK:          %[[SLICE2:.+]] = tensor.extract_slice %[[ARG6]][%[[I]]] [%[[SIZE]]] [1] : tensor<1024xf32> to tensor<?xf32>
// CHECK:          %[[FFT:.+]]:2 = linalg_ext.fft
// CHECK-SAME:       {__internal_linalg_transform__ = "tiling_1d_stage5_fft_output"}
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME:       outs(%[[SLICE1]], %[[SLICE2]] : tensor<?xf32>, tensor<?xf32>)
// CHECK:          %[[INSERT1:.+]] = tensor.insert_slice %[[FFT]]#0 into %[[ARG5]][%[[I]]] [%[[SIZE]]] [1] : tensor<?xf32> into tensor<1024xf32>
// CHECK:          %[[INSERT2:.+]] = tensor.insert_slice %[[FFT]]#1 into %[[ARG6]][%[[I]]] [%[[SIZE]]] [1] : tensor<?xf32> into tensor<1024xf32>
// CHECK:          scf.yield %[[INSERT1]], %[[INSERT2]]
// CHECK:        return %[[RES]]#0, %[[RES]]#1 : tensor<1024xf32>, tensor<1024xf32>

// -----

func @fft_2d_stage_5(%arg0: tensor<3x1024xf32>, %arg1: tensor<3x1024xf32>,
    %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
  %cst1 = constant 5 : index
  %0:2 = linalg_ext.fft
  {__internal_linalg_transform__ = "tiling_2d_stage5_fft_input"}
    ins(%cst1, %arg2, %arg3: index, tensor<16xf32>, tensor<16xf32>)
    outs(%arg0, %arg1: tensor<3x1024xf32>, tensor<3x1024xf32>)
  : tensor<3x1024xf32>, tensor<3x1024xf32>
  return %0#0, %0#1 : tensor<3x1024xf32>, tensor<3x1024xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK:      func @fft_2d_stage_5(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = constant 0 : index
// CHECK-DAG:    %[[C3:.+]] = constant 3 : index
// CHECK-DAG:    %[[C5:.+]] = constant 5 : index
// CHECK-DAG:    %[[C10:.+]] = constant 10 : index
// CHECK-DAG:    %[[C32:.+]] = constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = constant 1024 : index
// CHECK:        %[[RES:.+]]:2 = scf.for %[[I:.+]] = %[[C0]] to %[[C3]] step %[[C10]]
// CHECK-SAME:       iter_args(%[[ARG5:.+]] = %[[ARG0]], %[[ARG6:.+]] = %[[ARG1]])
// CHECK-SAME:       -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
// CHECK:          %[[SZ1:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C10]], %[[C3]]]
// CHECK:          %{{.+}} = scf.for %[[J:.+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:         iter_args(%[[ARG8:.+]] = %[[ARG5]], %[[ARG9:.+]] = %[[ARG6]]) -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
// CHECK:            %[[SZ2:.+]] = affine.min #[[MAP1]](%[[J]])[%[[C32]], %[[C1024]]]
// CHECK:            %[[SLICE1:.+]] = tensor.extract_slice %[[ARG8]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:            %[[SLICE2:.+]] = tensor.extract_slice %[[ARG9]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:          %[[FFT:.+]]:2 = linalg_ext.fft
// CHECK-SAME:       {__internal_linalg_transform__ = "tiling_2d_stage5_fft_output"}
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME:       outs(%[[SLICE1]], %[[SLICE2]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK:          %[[INSERT1:.+]] = tensor.insert_slice %[[FFT]]#0 into %[[ARG8]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:          %[[INSERT2:.+]] = tensor.insert_slice %[[FFT]]#1 into %[[ARG9]][%[[I]], %[[J]]] [%[[SZ1]], %[[SZ2]]] [1, 1]
// CHECK:          scf.yield %[[INSERT1]], %[[INSERT2]] : tensor<3x1024xf32>, tensor<3x1024xf32>

// -----

func @fft_1d_stage_5_memref(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>,
    %arg2: memref<16xf32>, %arg3: memref<16xf32>) {
  %cst1 = constant 5 : index
  linalg_ext.fft
  {__internal_linalg_transform__ = "tiling_1d_stage5_fft_input"}
    ins(%cst1, %arg2, %arg3: index, memref<16xf32>, memref<16xf32>)
    outs(%arg0, %arg1: memref<1024xf32>, memref<1024xf32>)
  return
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK:      func @fft_1d_stage_5_memref(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = constant 0 : index
// CHECK-DAG:    %[[C5:.+]] = constant 5 : index
// CHECK-DAG:    %[[C32:.+]] = constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = constant 1024 : index
// CHECK:        scf.for %[[I:.+]] = %[[C0]] to %[[C1024]] step %[[C32]] {
// CHECK:          %[[SZ:.+]] = affine.min #[[MAP0]](%[[I]])[%[[C32]], %[[C1024]]]
// CHECK:          %[[SUB1:.+]] = memref.subview %[[ARG0]][%[[I]]] [%[[SZ]]] [1] : memref<1024xf32> to memref<?xf32, #[[MAP1]]>
// CHECK:          %[[SUB2:.+]] = memref.subview %[[ARG1]][%[[I]]] [%[[SZ]]] [1] : memref<1024xf32> to memref<?xf32, #[[MAP1]]>
// CHECK:          linalg_ext.fft
// CHECK-SAME:       {__internal_linalg_transform__ = "tiling_1d_stage5_fft_output"}
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, memref<16xf32>, memref<16xf32>)
// CHECK-SAME:       outs(%[[SUB1]], %[[SUB2]] : memref<?xf32, #[[MAP1]]>, memref<?xf32, #[[MAP1]]>)
