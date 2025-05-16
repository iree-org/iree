// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --verify-diagnostics -canonicalize -cse %s | FileCheck  %s

func.func @scatter_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
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
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
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
//       CHECK:     %[[RESULT_INNER:.+]] = scf.for %[[IV1:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZEX]]
//  CHECK-SAME:         iter_args(%[[INITX:.+]] = %[[INITY]])
//   CHECK-DAG:     %[[USED_TILESIZEY:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[D0]]]
//   CHECK-DAG:       %[[USED_TILESIZEX:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[D1]]]
//       CHECK:       %[[UPDATE_SLICE:.+]] = tensor.extract_slice %[[UPDATES]][%[[IV0]], %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], %[[USED_TILESIZEX]]]
//       CHECK:       %[[INDEX_SLICE:.+]] = tensor.extract_slice %[[INDICES]][%[[IV0]], 0]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], 1]
//       CHECK:       %[[SCATTER_DIM:.+]] = tensor.dim %[[INITX]], %[[C0]]
//       CHECK:       %[[ORIGINAL_SLICE:.+]] = tensor.extract_slice %[[INITX]][0, %[[IV1]]]
//  CHECK-SAME:           [%[[SCATTER_DIM]], %[[USED_TILESIZEX]]]
//       CHECK:       %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:           unique_indices(true)
//  CHECK-SAME:           ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:           outs(%[[ORIGINAL_SLICE]]
//       CHECK:       %[[SCATTER_DIM2:.+]] = tensor.dim %[[ORIGINAL]], %[[C0]]
//       CHECK:       %[[YIELD:.+]] = tensor.insert_slice %[[SCATTER_TILE]] into %[[INITX]][0, %[[IV1]]]
//  CHECK-SAME:           [%[[SCATTER_DIM2]], %[[USED_TILESIZEX]]]
//       CHECK:       scf.yield %[[YIELD]]
//       CHECK:     scf.yield %[[RESULT_INNER]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_tiling_memref(
    %original: memref<?x?xf32>, %indices: memref<?x1xi32>,
    %update : memref<?x?xf32>) {
  iree_linalg_ext.scatter
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
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
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
//       CHECK:     scf.for %[[IV1:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZEX]]
//   CHECK-DAG:       %[[USED_TILESIZEY:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[D0]]]
//   CHECK-DAG:       %[[USED_TILESIZEX:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[D1]]]
//       CHECK:       %[[UPDATE_SLICE:.+]] = memref.subview %[[UPDATES]][%[[IV0]], %[[IV1]]]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], %[[USED_TILESIZEX]]]
//       CHECK:       %[[INDEX_SLICE:.+]] = memref.subview %[[INDICES]][%[[IV0]], 0]
//  CHECK-SAME:           [%[[USED_TILESIZEY]], 1]
//       CHECK:       %[[SCATTER_DIM:.+]] = memref.dim %[[ORIGINAL]], %[[C0]]
//       CHECK:       %[[ORIGINAL_SLICE:.+]] = memref.subview %[[ORIGINAL]][0, %[[IV1]]
//  CHECK-SAME:           [%[[SCATTER_DIM]], %[[USED_TILESIZEX]]]
//       CHECK:       iree_linalg_ext.scatter
//  CHECK-SAME:           unique_indices(true)
//  CHECK-SAME:           ins(%[[UPDATE_SLICE]], %[[INDEX_SLICE]]
//  CHECK-SAME:           outs(%[[ORIGINAL_SLICE]]

// -----

func.func @scatter_no_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
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
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.tile_using_for %0 tile_sizes [0] : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
//       CHECK: func.func @scatter_no_tiling
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATES:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:       unique_indices(true)
//  CHECK-SAME:       ins(%[[UPDATES]], %[[INDICES]]
//  CHECK-SAME:       outs(%[[ORIGINAL]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_repeated_indices_tiling(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
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
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
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
//       CHECK:     %[[SZ:.+]] = affine.min #[[MAP]](%[[I]])[%[[D1]]]
//       CHECK:       %[[UPDATES_TILE:.+]] = tensor.extract_slice
//  CHECK-SAME:         %[[UPDATES]][0, %[[I]]] [%[[D0]], %[[SZ]]] [1, 1]
//       CHECK:       %[[INDICES_TILE:.+]] = tensor.extract_slice
//  CHECK-SAME:         %[[INDICES]][0, 0] [%[[D0]], 1] [1, 1]
//       CHECK:       %[[ITER_D0:.+]] = tensor.dim %[[ITER]], %[[C0]]
//       CHECK:       %[[ORIGINAL_TILE:.+]] = tensor.extract_slice
//  CHECK-SAME:         %[[ITER]][0, %[[I]]] [%[[ITER_D0]], %[[SZ]]] [1, 1]
//       CHECK:       %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:         unique_indices(false)
//  CHECK-SAME:         ins(%[[UPDATES_TILE]], %[[INDICES_TILE]]
//  CHECK-SAME:         outs(%[[ORIGINAL_TILE]]
//       CHECK:       %[[ORIGINAL_D0:.+]] = tensor.dim %[[ORIGINAL]], %[[C0]]
//       CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SCATTER]] into
//  CHECK-SAME:         %[[ITER]][0, %[[I]]] [%[[ORIGINAL_D0]], %[[SZ]]] [1, 1]
//       CHECK:       scf.yield %[[RES]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_batch_2D(
    %original: memref<?xi32>, %indices: memref<?x?x1xi32>,
    %updates: memref<?x?xi32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates, %indices : memref<?x?xi32>, memref<?x?x1xi32>)
    outs(%original : memref<?xi32>)  {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  }
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
// CHECK:       func.func @scatter_batch_2D
// CHECK-SAME:    %[[ORIGINAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[UPDATES:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C20:.+]] = arith.constant 20 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[UPDATES]], %[[C0]]
// CHECK-DAG:     %[[D1:.+]] = memref.dim %[[UPDATES]], %[[C1]]
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[D1]] step %[[C20]]
// CHECK:           %[[SZ:.+]] = affine.min #[[MAP]](%[[I]])[%[[D1]]]
// CHECK:           %[[UPDATES_TILE:.+]] = memref.subview
// CHECK-SAME:             %[[UPDATES]][0, %[[I]]]
// CHECK-SAME:             [%[[D0]], %[[SZ]]]
// CHECK:           %[[INDICES_TILE:.+]] = memref.subview
// CHECK-SAME:             %[[INDICES]][0, %[[I]], 0]
// CHECK-SAME:             [%[[D0]], %[[SZ]], 1]
// CHECK:           %[[ORIGINAL_TILE:.+]] = memref.subview
// CHECK-SAME:             %[[ORIGINAL]][0]
// CHECK:           iree_linalg_ext.scatter
// CHECK-SAME:             unique_indices(true)
// CHECK-SAME:             ins(%[[UPDATES_TILE]], %[[INDICES_TILE]]
// CHECK-SAME:             outs(%[[ORIGINAL_TILE]]

// -----


func.func @sort_1d(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = iree_linalg_ext.sort
       dimension(0)
       outs(%arg0 : tensor<?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):
         %0 = arith.cmpi sgt, %arg2, %arg3 : i32
         iree_linalg_ext.yield %0 : i1
       } -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.tile_using_for %0 tile_sizes [0] : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @sort_1d(
// CHECK-SAME:   %[[OPERAND:.+]]: tensor<?xi32>
//      CHECK:   %[[RESULT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:       outs(%[[OPERAND]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @sort_2d(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = iree_linalg_ext.sort
       dimension(1)
       outs(%arg0 : tensor<?x?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):
         %0 = arith.cmpi sgt, %arg2, %arg3 : i32
         iree_linalg_ext.yield %0 : i1
       } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [10, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
//       CHECK: func.func @sort_2d(
//  CHECK-SAME:   %[[OPERAND:.+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 10 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[OPERAND]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[D0]]]
//       CHECK:     %[[OPERAND_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[SORT_TILE:.+]] = iree_linalg_ext.sort
//  CHECK-SAME:         outs(%[[OPERAND_SLICE]]
//       CHECK:     %[[YIELD:.+]] = tensor.insert_slice %[[SORT_TILE]] into %[[INIT]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     scf.yield %[[YIELD]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @sort_2d_inner_parallel(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = iree_linalg_ext.sort
       dimension(0)
       outs(%arg0 : tensor<?x?xi32>) {
       ^bb0(%arg2: i32, %arg3: i32):
         %0 = arith.cmpi sgt, %arg2, %arg3 : i32
         iree_linalg_ext.yield %0 : i1
       } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
//       CHECK: func.func @sort_2d_inner_parallel(
//  CHECK-SAME:   %[[OPERAND:.+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
//       CHECK:   %[[RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZE]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[OPERAND]])
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[D1]]]
//       CHECK:     %[[OPERAND_SLICE:.+]] = tensor.extract_slice %[[INIT]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     %[[SORT_TILE:.+]] = iree_linalg_ext.sort
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
       dimension(1)
       outs(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xf32>) {
       ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):
         %1 = arith.cmpf ogt, %arg4, %arg5 : f32
         iree_linalg_ext.yield %1 : i1
       } -> tensor<?x?xi32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [10, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
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
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[D0]]]
//       CHECK:     %[[OPERAND1_SLICE:.+]] = tensor.extract_slice %[[INIT1]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[OPERAND2_SLICE:.+]] = tensor.extract_slice %[[INIT2]][%[[IV]], 0]
//  CHECK-SAME:         [%[[USED_TILESIZE]], %[[D1]]]
//       CHECK:     %[[SORT_TILE:.+]]:2 = iree_linalg_ext.sort
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
     dimension(0)
     outs(%arg0, %arg1 : memref<?x?xi32>, memref<?x?xf32>) {
     ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):
       %0 = arith.cmpf ogt, %arg4, %arg5 : f32
       iree_linalg_ext.yield %0 : i1
     }
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %1, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
//       CHECK: func.func @sort_2d_multi_result_memref(
//  CHECK-SAME:   %[[OPERAND1:.+]]: memref<?x?xi32>
//  CHECK-SAME:   %[[OPERAND2:.+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[TILESIZE:.+]] = arith.constant 20 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[OPERAND1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[OPERAND1]], %[[C1]]
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[D1]] step %[[TILESIZE]]
//   CHECK-DAG:     %[[USED_TILESIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[D1]]]
//       CHECK:     %[[OPERAND1_SLICE:.+]] = memref.subview %[[OPERAND1]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     %[[OPERAND2_SLICE:.+]] = memref.subview %[[OPERAND2]][0, %[[IV]]]
//  CHECK-SAME:         [%[[D0]], %[[USED_TILESIZE]]]
//       CHECK:     iree_linalg_ext.sort
//  CHECK-SAME:         outs(%[[OPERAND1_SLICE]], %[[OPERAND2_SLICE]]

// -----

func.func @fft_1d_stage_5(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>,
    %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  %cst1 = arith.constant 5 : index
  %0:2 = iree_linalg_ext.fft
    ins(%cst1, %arg2, %arg3: index, tensor<16xf32>, tensor<16xf32>)
    outs(%arg0, %arg1: tensor<1024xf32>, tensor<1024xf32>)
  : tensor<1024xf32>, tensor<1024xf32>
  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.fft"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
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
// CHECK:          %[[SLICE1:.+]] = tensor.extract_slice %[[ARG5]][%[[I]]] [32] [1]
// CHECK:          %[[SLICE2:.+]] = tensor.extract_slice %[[ARG6]][%[[I]]] [32] [1]
// CHECK:          %[[FFT:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:       outs(%[[SLICE1]], %[[SLICE2]]
// CHECK:          %[[INSERT1:.+]] = tensor.insert_slice %[[FFT]]#0 into %[[ARG5]][%[[I]]] [32] [1]
// CHECK:          %[[INSERT2:.+]] = tensor.insert_slice %[[FFT]]#1 into %[[ARG6]][%[[I]]] [32] [1]
// CHECK:          scf.yield %[[INSERT1]], %[[INSERT2]]
// CHECK:        return %[[RES]]#0, %[[RES]]#1 : tensor<1024xf32>, tensor<1024xf32>

// -----

func.func @fft_2d_stage_5(%arg0: tensor<3x1024xf32>, %arg1: tensor<3x1024xf32>,
    %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
  %cst1 = arith.constant 5 : index
  %0:2 = iree_linalg_ext.fft
    ins(%cst1, %arg2, %arg3: index, tensor<16xf32>, tensor<16xf32>)
    outs(%arg0, %arg1: tensor<3x1024xf32>, tensor<3x1024xf32>)
  : tensor<3x1024xf32>, tensor<3x1024xf32>
  return %0#0, %0#1 : tensor<3x1024xf32>, tensor<3x1024xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.fft"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK:      func.func @fft_2d_stage_5(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[RES:.+]]:2 = scf.for %[[J:.+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG8:.+]] = %[[ARG0]], %[[ARG9:.+]] = %[[ARG1]]) -> (tensor<3x1024xf32>, tensor<3x1024xf32>) {
// CHECK:          %[[SLICE1:.+]] = tensor.extract_slice %[[ARG8]][0, %[[J]]] [3, 32] [1, 1]
// CHECK:          %[[SLICE2:.+]] = tensor.extract_slice %[[ARG9]][0, %[[J]]] [3, 32] [1, 1]
// CHECK:        %[[FFT:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:     ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:     outs(%[[SLICE1]], %[[SLICE2]]
// CHECK:        %[[INSERT1:.+]] = tensor.insert_slice %[[FFT]]#0 into %[[ARG8]][0, %[[J]]] [3, 32] [1, 1]
// CHECK:        %[[INSERT2:.+]] = tensor.insert_slice %[[FFT]]#1 into %[[ARG9]][0, %[[J]]] [3, 32] [1, 1]
// CHECK:        scf.yield %[[INSERT1]], %[[INSERT2]] : tensor<3x1024xf32>, tensor<3x1024xf32>

// -----

func.func @fft_1d_stage_5_memref(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>,
    %arg2: memref<16xf32>, %arg3: memref<16xf32>) {
  %cst1 = arith.constant 5 : index
  iree_linalg_ext.fft
    ins(%cst1, %arg2, %arg3: index, memref<16xf32>, memref<16xf32>)
    outs(%arg0, %arg1: memref<1024xf32>, memref<1024xf32>)
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.fft"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
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
// CHECK:          %[[SUB1:.+]] = memref.subview %[[ARG0]][%[[I]]] [32] [1]
// CHECK:          %[[SUB2:.+]] = memref.subview %[[ARG1]][%[[I]]] [32] [1]
// CHECK:          iree_linalg_ext.fft
// CHECK-SAME:       ins(%[[C5]], %[[COEF_REAL]], %[[COEF_IMAG]]
// CHECK-SAME:       outs(%[[SUB1]], %[[SUB2]]

// -----

func.func @scan_1d(%0: tensor<128xi32>) -> tensor<128xi32> {
  %c0 = tensor.empty() : tensor<i32>
  %1 = tensor.empty() : tensor<128xi32>
  %2:2 = iree_linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<128xi32>) outs(%1, %c0 : tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %2#0 : tensor<128xi32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scan"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.tile_using_for %0 tile_sizes [0] : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @scan_1d(
// CHECK-SAME:   %[[OPERAND:.+]]: tensor<128xi32>
//      CHECK:   %[[ACC:.+]] = tensor.empty() : tensor<i32>
//      CHECK:   %[[OUTPUT:.+]] = tensor.empty() : tensor<128xi32>
//      CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.scan
// CHECK-SAME:       ins(%[[OPERAND]] :
// CHECK-SAME:       outs(%[[OUTPUT]], %[[ACC]] :
//      CHECK:   return %[[RESULT]]

// -----

func.func @scan_2d(%0: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %c0 = tensor.empty() : tensor<32xi32>
  %1 = tensor.empty() : tensor<16x32xi32>
  %2:2 = iree_linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<16x32xi32>) outs(%1, %c0 : tensor<16x32xi32>, tensor<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  } -> tensor<16x32xi32>, tensor<32xi32>
  return %2#0 : tensor<16x32xi32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scan"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0) -> (-d0 + 32, 20)>
//      CHECK:  func.func @scan_2d(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG:    %[[C20:.+]] = arith.constant 20 : index
//  CHECK-DAG:    %[[ACC:.+]] = tensor.empty() : tensor<32xi32>
//  CHECK-DAG:    %[[OUTPUT:.+]] = tensor.empty() : tensor<16x32xi32>
//      CHECK:    %[[RESULT:.+]]:2 = scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C20]]
// CHECK-SAME:      iter_args(%[[ARG2:.+]] = %[[OUTPUT]], %[[ARG3:.+]] = %[[ACC]])
//      CHECK:      %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])
//      CHECK:      %[[UPDATE_SLICE_IN:.+]] = tensor.extract_slice %[[ARG0]][0, %[[I]]] [16, %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_OUT:.+]] = tensor.extract_slice %[[ARG2]][0, %[[I]]] [16, %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_ACC:.+]] = tensor.extract_slice %[[ARG3]][%[[I]]] [%[[SIZE]]]
//      CHECK:      %[[SCAN_TILE:.+]]:2 = iree_linalg_ext.scan
// CHECK-SAME:       dimension(0) inclusive(true)
// CHECK-SAME:       ins(%[[UPDATE_SLICE_IN]]
// CHECK-SAME:       outs(%[[UPDATE_SLICE_OUT]], %[[UPDATE_SLICE_ACC]]
//      CHECK:       %[[YIELD:.+]] = tensor.insert_slice %[[SCAN_TILE]]#0 into %[[ARG2]][0, %[[I]]]
// CHECK-SAME:           [16, %[[SIZE]]]
//      CHECK:       %[[ACC_YIELD:.+]] = tensor.insert_slice %[[SCAN_TILE]]#1 into %[[ARG3]][%[[I]]]
// CHECK-SAME:           [%[[SIZE]]]
//      CHECK:       scf.yield %[[YIELD]], %[[ACC_YIELD]] : tensor<16x32xi32>, tensor<32xi32>
//      CHECK:   return %[[RESULT]]#0

// -----

func.func @scan_2d_memref(%0: memref<16x32xi32>, %1: memref<16x32xi32>) {
  %c0 = memref.alloc() : memref<32xi32>
  iree_linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : memref<16x32xi32>) outs(%1, %c0 : memref<16x32xi32>, memref<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      iree_linalg_ext.yield %sum : i32
  }
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.scan"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK:  #[[MAP0:.+]] = affine_map<(d0) -> (-d0 + 32, 20)>
//      CHECK:  func.func @scan_2d_memref(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
//  CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG:    %[[C20:.+]] = arith.constant 20 : index
//  CHECK-DAG:    %[[ACC:.+]] = memref.alloc() : memref<32xi32>
//      CHECK:    scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C20]]
//      CHECK:      %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])
//      CHECK:      %[[UPDATE_SLICE_IN:.+]] = memref.subview %[[ARG0]][0, %[[I]]] [16, %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_OUT:.+]] = memref.subview %[[ARG1]][0, %[[I]]] [16, %[[SIZE]]]
//      CHECK:      %[[UPDATE_SLICE_ACC:.+]] = memref.subview %[[ACC]][%[[I]]] [%[[SIZE]]]
//      CHECK:      iree_linalg_ext.scan
// CHECK-SAME:       dimension(0) inclusive(true)
// CHECK-SAME:       ins(%[[UPDATE_SLICE_IN]]
// CHECK-SAME:       outs(%[[UPDATE_SLICE_OUT]], %[[UPDATE_SLICE_ACC]]
//      CHECK:   return

// -----

func.func @topk_tile_tensor(%input_values: tensor<?x?xf32>, %input_indices: tensor<?x?xi32>, %out_values: tensor<?x3xf32> , %out_indices: tensor<?x3xi32>) -> (tensor<?x3xf32>, tensor<?x3xi32>) {
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<?x?xf32> , tensor<?x?xi32>)
        outs(%out_values, %out_indices : tensor<?x3xf32>, tensor<?x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<?x3xf32>, tensor<?x3xi32>
  return %0#0, %0#1 : tensor<?x3xf32>, tensor<?x3xi32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.topk"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [10, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
// CHECK:       func.func @topk_tile_tensor
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
// CHECK:           %[[D3:.+]] = affine.min #[[MAP0]](%[[ARG4]])[%[[D0]]]
// CHECK:           %[[D4:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG4]], 0] [%[[D3]], %[[D1]]] [1, 1]
// CHECK:           %[[D5:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0] [%[[D3]], %[[D1]]] [1, 1]
// CHECK:           %[[D6:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG4]], 0] [%[[D3]], 3] [1, 1]
// CHECK:           %[[D7:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG4]], 0] [%[[D3]], 3] [1, 1]
// CHECK:           %[[D8:.+]]:2 = iree_linalg_ext.topk
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
        dimension(1)
        ins(%input_values, %input_indices : memref<?x?xf32> , memref<?x?xi32>)
        outs(%out_values, %out_indices : memref<?x3xf32>, memref<?x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        }
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.topk"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [10, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK:       #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
// CHECK:       func.func @topk_tile_memref
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
// CHECK:           %[[D2:.+]] = affine.min #[[MAP0]](%[[ARG4]])[%[[D0]]]
// CHECK:           %[[D3:.+]] = memref.subview %[[ARG0]][%[[ARG4]], 0] [%[[D2]], %[[D1]]] [1, 1]
// CHECK:           %[[D4:.+]] = memref.subview %[[ARG1]][%[[ARG4]], 0] [%[[D2]], %[[D1]]] [1, 1]
// CHECK:           %[[D5:.+]] = memref.subview %[[ARG2]][%[[ARG4]], 0] [%[[D2]], 3] [1, 1]
// CHECK:           %[[D6:.+]] = memref.subview %[[ARG3]][%[[ARG4]], 0] [%[[D2]], 3] [1, 1]
// CHECK:           iree_linalg_ext.topk
// CHECK-SAME:        dimension(1)
// CHECK-SAME:        ins(%[[D3]], %[[D4]]
// CHECK-SAME:        outs(%[[D5]], %[[D6]]
// CHECK:           return

// -----

func.func @topk_tile_tensor_optional(%input_values: tensor<20x10xf32>, %out_values: tensor<20x3xf32> , %out_indices: tensor<20x3xi32>) -> (tensor<20x3xf32>, tensor<20x3xi32>) {
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values : tensor<20x10xf32>)
        outs(%out_values, %out_indices : tensor<20x3xf32>, tensor<20x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<20x3xf32>, tensor<20x3xi32>
  return %0#0, %0#1 : tensor<20x3xf32>, tensor<20x3xi32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.topk"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [10, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @topk_tile_tensor_optional
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C20:.+]] = arith.constant 20 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[RESULT:.+]]:2 = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C20]] step %[[C10]] iter_args(%[[ARG4:.+]] = %[[ARG1]], %[[ARG5:.+]] = %[[ARG2]])
// CHECK:           %[[D2:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0] [10, 10] [1, 1]
// CHECK:           %[[D3:.+]] = tensor.extract_slice %[[ARG4]][%[[ARG3]], 0] [10, 3] [1, 1]
// CHECK:           %[[D4:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG3]], 0] [10, 3] [1, 1]
// CHECK:           %[[D5:.+]]:2 = iree_linalg_ext.topk
// CHECK-SAME:        dimension(1)
// CHECK-SAME:        ins(%[[D2]]
// CHECK-SAME:        outs(%[[D3]], %[[D4]]
// CHECK:           %[[D6:.+]] = tensor.insert_slice %[[D5]]#0 into %[[ARG4]][%[[ARG3]], 0] [10, 3] [1, 1]
// CHECK:           %[[D7:.+]] = tensor.insert_slice %[[D5]]#1 into %[[ARG5]][%[[ARG3]], 0] [10, 3] [1, 1]
// CHECK:           scf.yield %[[D6]], %[[D7]]
// CHECK:           return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @im2col(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [34] * [1] k_offset = [1000] * [1]
           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           input_k_perm = [0, 1, 2]
           ins(%arg0 : tensor<2x34x34x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.im2col"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [1, 5, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (-d0 + 1024, 5)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 + 1000)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 + 34)>
// CHECK:      func.func @im2col(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>) -> tensor<2x1024x5760xf32>
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:    %[[C5760:.+]] = arith.constant 5760 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<2x1024x5760xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<2x1024x5760xf32>)
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C5]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<2x1024x5760xf32>)
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C5760]] step %[[C4]]
// CHECK-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<2x1024x5760xf32>)
// CHECK-DAG:          %[[MSIZE:.+]] = affine.min #[[MAP]](%[[ARG3]])
// CHECK-DAG:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, 0]
// CHECK-SAME:           [1, 34, 34, 640] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x34x34x640xf32>
// CHECK-DAG:          %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG1]], %[[ARG3]], %[[ARG5]]]
// CHECK-SAME:           [1, %[[MSIZE]], 4] [1, 1, 1] : tensor<2x1024x5760xf32> to tensor<1x?x4xf32>
// CHECK-DAG:          %[[KOFFSET:.+]] = affine.apply #[[MAP1]](%[[ARG5]])
// CHECK-DAG:          %[[MOFFSET:.+]] = affine.apply #[[MAP2]](%[[ARG3]])
// CHECK:              %[[IM2COL:.+]] = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:           m_offset = [%[[MOFFSET]]] * [1] k_offset = [%[[KOFFSET]]] * [1]
// CHECK-SAME:           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:           input_k_perm = [0, 1, 2]
// CHECK-SAME:           ins(%[[EXTRACTED_SLICE]] : tensor<1x34x34x640xf32>)
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_0]] : tensor<1x?x4xf32>) -> tensor<1x?x4xf32>
// CHECK:              %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[IM2COL]] into %[[ARG6]]
// CHECK-SAME:           [%[[ARG1]], %[[ARG3]], %[[ARG5]]] [1, %[[MSIZE]], 4] [1, 1, 1]
// CHECK-SAME:           tensor<1x?x4xf32> into tensor<2x1024x5760xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE]] : tensor<2x1024x5760xf32>
// CHECK:            scf.yield %[[RES2]] : tensor<2x1024x5760xf32>
// CHECK:          scf.yield %[[RES1]] : tensor<2x1024x5760xf32>
// CHECK:        return %[[RES0]] : tensor<2x1024x5760xf32>

// -----

func.func @im2col_transposed_m_pos(%arg0: tensor<640x2x101x172xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  %1 = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
           m_offset = [42] * [1] k_offset = [7] * [1]
           batch_pos = [1] m_pos = [3, 2] k_pos = [0]
           input_k_perm = [0, 1, 2]
           ins(%arg0 : tensor<640x2x101x172xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.im2col"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [1, 9, 7] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (-d0 + 1024, 9)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 5760, 7)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 + 7)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (d0 + 42)>
// CHECK:      func.func @im2col_transposed_m_pos(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<640x2x101x172xf32>) -> tensor<2x1024x5760xf32>
// CHECK-DAG:    %[[C7:.+]] = arith.constant 7 : index
// CHECK-DAG:    %[[C9:.+]] = arith.constant 9 : index
// CHECK-DAG:    %[[C5760:.+]] = arith.constant 5760 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<2x1024x5760xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<2x1024x5760xf32>)
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C9]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<2x1024x5760xf32>)
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C5760]] step %[[C7]]
// CHECK-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<2x1024x5760xf32>)
// CHECK-DAG:          %[[MSIZE:.+]] = affine.min #[[MAP]](%[[ARG3]])
// CHECK-DAG:          %[[KSIZE:.+]] = affine.min #[[MAP1]](%[[ARG5]])
// CHECK-DAG:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[ARG1]], 0, 0]
// CHECK-SAME:           [640, 1, 101, 172] [1, 1, 1, 1] : tensor<640x2x101x172xf32> to tensor<640x1x101x172xf32>
// CHECK-DAG:          %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG1]], %[[ARG3]], %[[ARG5]]]
// CHECK-SAME:           [1, %[[MSIZE]], %[[KSIZE]]] [1, 1, 1] : tensor<2x1024x5760xf32> to tensor<1x?x?xf32>
// CHECK-DAG:          %[[KOFFSET:.+]] = affine.apply #[[MAP2]](%[[ARG5]])
// CHECK-DAG:          %[[MOFFSET:.+]] = affine.apply #[[MAP3]](%[[ARG3]])
// CHECK:              %[[IM2COL:.+]] = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
// CHECK-SAME:           m_offset = [%[[MOFFSET]]] * [1] k_offset = [%[[KOFFSET]]] * [1]
// CHECK-SAME:           batch_pos = [1] m_pos = [3, 2] k_pos = [0]
// CHECK-SAME:           input_k_perm = [0, 1, 2]
// CHECK-SAME:           ins(%[[EXTRACTED_SLICE]] : tensor<640x1x101x172xf32>)
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_0]] : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// CHECK:              %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[IM2COL]] into %[[ARG6]]
// CHECK-SAME:           [%[[ARG1]], %[[ARG3]], %[[ARG5]]] [1, %[[MSIZE]], %[[KSIZE]]] [1, 1, 1]
// CHECK-SAME:           tensor<1x?x?xf32> into tensor<2x1024x5760xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE]] : tensor<2x1024x5760xf32>
// CHECK:            scf.yield %[[RES2]] : tensor<2x1024x5760xf32>
// CHECK:          scf.yield %[[RES1]] : tensor<2x1024x5760xf32>
// CHECK:        return %[[RES0]] : tensor<2x1024x5760xf32>

// -----

func.func @im2col_dynamic(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index,
                          %mOffset: index, %kOffset: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%s0, %s1, %s2) : tensor<?x?x?xf32>
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [%mOffset] * [1] k_offset = [%kOffset] * [1]
           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           input_k_perm = [0, 1, 2]
           ins(%arg0 : tensor<?x?x?x?xf32>)
           outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.im2col"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [2, 7, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 7)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 5)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK:      func.func @im2col_dynamic(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:   %[[S0:.+]]: index, %[[S1:.+]]: index, %[[S2:.+]]: index, %[[MOFF:.+]]: index, %[[KOFF:.+]]: index
// CHECK-DAG:    %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:    %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:    %[[C7:.+]] = arith.constant 7 : index
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = tensor.empty(%[[S0]], %[[S1]], %[[S2]]) : tensor<?x?x?xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S0]] step %[[C2]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<?x?x?xf32>)
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S1]] step %[[C7]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<?x?x?xf32>)
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S2]] step %[[C5]]
// CHECK-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<?x?x?xf32>)
// CHECK-DAG:          %[[BSIZE:.+]] = affine.min #[[MAP]](%[[ARG1]])
// CHECK-DAG:          %[[MSIZE:.+]] = affine.min #[[MAP1]](%[[ARG3]])
// CHECK-DAG:          %[[KSIZE:.+]] = affine.min #[[MAP2]](%[[ARG5]])
// CHECK-DAG:          %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK-DAG:          %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK-DAG:          %[[DIM3:.+]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK-DAG:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, 0]
// CHECK-SAME:           [%[[BSIZE]], %[[DIM1]], %[[DIM2]], %[[DIM3]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
// CHECK-DAG:          %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG1]], %[[ARG3]], %[[ARG5]]]
// CHECK-SAME:           [%[[BSIZE]], %[[MSIZE]], %[[KSIZE]]] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
// CHECK-DAG:          %[[KOFFSET:.+]] = affine.apply #[[MAP3]](%[[ARG5]])[%[[KOFF]]]
// CHECK-DAG:          %[[MOFFSET:.+]] = affine.apply #[[MAP3]](%[[ARG3]])[%[[MOFF]]]
// CHECK:              %[[IM2COL:.+]] = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:           m_offset = [%[[MOFFSET]]] * [1] k_offset = [%[[KOFFSET]]] * [1]
// CHECK-SAME:           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:           input_k_perm = [0, 1, 2]
// CHECK-SAME:           ins(%[[EXTRACTED_SLICE]] : tensor<?x?x?x?xf32>)
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_0]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:              %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[IM2COL]] into %[[ARG6]]
// CHECK-SAME:           [%[[ARG1]], %[[ARG3]], %[[ARG5]]] [%[[BSIZE]], %[[MSIZE]], %[[KSIZE]]] [1, 1, 1]
// CHECK-SAME:           tensor<?x?x?xf32> into tensor<?x?x?xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE]] : tensor<?x?x?xf32>
// CHECK:            scf.yield %[[RES2]] : tensor<?x?x?xf32>
// CHECK:          scf.yield %[[RES1]] : tensor<?x?x?xf32>
// CHECK:        return %[[RES0]] : tensor<?x?x?xf32>

// -----

module {
  func.func @im2col_expanded(%arg0: tensor<2x34x34x640xf32>, %m_stride: index) -> tensor<2x32x32x1440x4xf32> {
    %0 = tensor.empty() : tensor<2x32x32x1440x4xf32>
    %7 = iree_linalg_ext.im2col
            strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
            m_offset = [0, 0] * [%m_stride, 1] k_offset = [0, 0] * [4, 1]
            batch_pos = [0] m_pos = [1, 2] k_pos = [3]
            input_k_perm = [0, 1, 2]
            ins(%arg0 : tensor<2x34x34x640xf32>)
            outs(%0 : tensor<2x32x32x1440x4xf32>) -> tensor<2x32x32x1440x4xf32>
    return %7 : tensor<2x32x32x1440x4xf32>
  }
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.im2col"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:5 = transform.structured.tile_using_for %0 tile_sizes [1, 7, 5, 11, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (-d0 + 32, 7)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 32, 5)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (-d0 + 1440, 11)>
// CHECK:      func.func @im2col_expanded
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
// CHECK-SAME:   %[[M_STRIDE:[a-zA-Z0-9_]+]]: index
// CHECK-DAG:    %[[C11:.+]] = arith.constant 11 : index
// CHECK-DAG:    %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:    %[[C7:.+]] = arith.constant 7 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[C1440:.+]] = arith.constant 1440 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<2x32x32x1440x4xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<2x32x32x1440x4xf32>)
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C7]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<2x32x32x1440x4xf32>)
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C5]]
// CHECK-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<2x32x32x1440x4xf32>)
// CHECK:              %[[RES3:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1440]] step %[[C11]]
// CHECK-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<2x32x32x1440x4xf32>)
// CHECK:                %[[RES4:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C2]]
// CHECK-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<2x32x32x1440x4xf32>)
// CHECK-DAG:              %[[M0SIZE:.+]] = affine.min #[[MAP]](%[[ARG3]])
// CHECK-DAG:              %[[M1SIZE:.+]] = affine.min #[[MAP1]](%[[ARG5]])
// CHECK-DAG:              %[[K0SIZE:.+]] = affine.min #[[MAP2]](%[[ARG7]])
// CHECK-DAG:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, 0]
// CHECK-SAME:               [1, 34, 34, 640] [1, 1, 1, 1] : tensor<2x34x34x640xf32> to tensor<1x34x34x640xf32>
// CHECK-DAG:              %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG10]][%[[ARG1]], %[[ARG3]], %[[ARG5]], %[[ARG7]], %[[ARG9]]]
// CHECK-SAME:               [1, %[[M0SIZE]], %[[M1SIZE]], %[[K0SIZE]], 2] [1, 1, 1, 1, 1] : tensor<2x32x32x1440x4xf32> to tensor<1x?x?x?x2xf32>
// CHECK:                  %[[IM2COL:.+]] = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:               m_offset = [%[[ARG3]], %[[ARG5]]] * [%[[M_STRIDE]], 1] k_offset = [%[[ARG7]], %[[ARG9]]] * [4, 1]
// CHECK-SAME:               batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:               input_k_perm = [0, 1, 2]
// CHECK-SAME:               ins(%[[EXTRACTED_SLICE]] : tensor<1x34x34x640xf32>)
// CHECK-SAME:               outs(%[[EXTRACTED_SLICE_0]] : tensor<1x?x?x?x2xf32>) -> tensor<1x?x?x?x2xf32>
// CHECK:                  %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[IM2COL]] into %[[ARG10]]
// CHECK-SAME:               [%[[ARG1]], %[[ARG3]], %[[ARG5]], %[[ARG7]], %[[ARG9]]] [1, %[[M0SIZE]], %[[M1SIZE]], %[[K0SIZE]], 2] [1, 1, 1, 1, 1]
// CHECK-SAME:               tensor<1x?x?x?x2xf32> into tensor<2x32x32x1440x4xf32>
// CHECK:                  scf.yield %[[INSERTED_SLICE]] : tensor<2x32x32x1440x4xf32>
// CHECK:                scf.yield %[[RES4]] : tensor<2x32x32x1440x4xf32>
// CHECK:              scf.yield %[[RES3]] : tensor<2x32x32x1440x4xf32>
// CHECK:            scf.yield %[[RES2]] : tensor<2x32x32x1440x4xf32>
// CHECK:          scf.yield %[[RES1]] : tensor<2x32x32x1440x4xf32>
// CHECK:        return %[[RES0]] : tensor<2x32x32x1440x4xf32>

// -----

func.func @winograd_filter_transform(%arg0: tensor<3x3x64x128xf32>) -> tensor<8x8x64x128xf32> {
  %0 = tensor.empty() : tensor<8x8x64x128xf32>
  %1 = iree_linalg_ext.winograd.filter_transform
    output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
    ins(%arg0 : tensor<3x3x64x128xf32>) outs(%0 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %1 : tensor<8x8x64x128xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.filter_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK:      func.func @winograd_filter_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<3x3x64x128xf32>) ->
// CHECK-SAME:   tensor<8x8x64x128xf32> {
// CHECK-DAG:    %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x64x128xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C64]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x64x128xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x64x128xf32>) {
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], %[[ARG3]]]
// CHECK-SAME:        [3, 3, 1, 1] [1, 1, 1, 1] : tensor<3x3x64x128xf32> to tensor<3x3x1x1xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG4]][0, 0, %[[ARG1]], %[[ARG3]]]
// CHECK-SAME:         [8, 8, 1, 1] [1, 1, 1, 1] : tensor<8x8x64x128xf32> to tensor<8x8x1x1xf32>
// CHECK:            %[[TF:.+]] = iree_linalg_ext.winograd.filter_transform
// CHECK-SAME:         output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
// CHECK-SAME:         ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE_0]]
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG4]]
// CHECK-SAME:         [0, 0, %[[ARG1]], %[[ARG3]]] [8, 8, 1, 1] [1, 1, 1, 1]
// CHECK-SAME:         tensor<8x8x1x1xf32> into tensor<8x8x64x128xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x64x128xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<8x8x64x128xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<8x8x64x128xf32>
// CHECK:      }

// -----

func.func @winograd_filter_transform_memref(%arg0: memref<3x3x64x128xf32>, %arg1: memref<8x8x64x128xf32>) {
  iree_linalg_ext.winograd.filter_transform
    output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
    ins(%arg0 : memref<3x3x64x128xf32>) outs(%arg1 : memref<8x8x64x128xf32>)
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.filter_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK:      func.func @winograd_filter_transform_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<3x3x64x128xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<8x8x64x128xf32>) {
// CHECK-DAG:    %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:          scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C128]] step %[[C1]] {
// CHECK:            %[[SUBVIEW:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:         [0, 0, %[[ARG2]], %[[ARG3]]] [3, 3, 1, 1] [1, 1, 1, 1]
// CHECK:            %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:         [0, 0, %[[ARG2]], %[[ARG3]]] [8, 8, 1, 1] [1, 1, 1, 1]
// CHECK:            iree_linalg_ext.winograd.filter_transform
// CHECK-SAME:         output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
// CHECK-SAME:         ins(%[[SUBVIEW]]
// CHECK-SAME:         outs(%[[SUBVIEW_0]]
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }

// -----

func.func @winograd_filter_transform_dynamic(%arg0: tensor<3x3x?x?xf32>, %s0: index, %s1: index) -> tensor<8x8x?x?xf32> {
  %0 = tensor.empty(%s0, %s1) : tensor<8x8x?x?xf32>
  %1 = iree_linalg_ext.winograd.filter_transform
    output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
    ins(%arg0 : tensor<3x3x?x?xf32>) outs(%0 : tensor<8x8x?x?xf32>) -> tensor<8x8x?x?xf32>
  return %1 : tensor<8x8x?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.filter_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK:      func.func @winograd_filter_transform_dynamic(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<3x3x?x?xf32>
// CHECK-SAME:   %[[S0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S1:[a-zA-Z0-9_]+]]: index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.empty(%[[S0]], %[[S1]]) : tensor<8x8x?x?xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S0]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x?x?xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S1]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x?x?xf32>) {
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], %[[ARG3]]]
// CHECK-SAME:        [3, 3, 1, 1] [1, 1, 1, 1] : tensor<3x3x?x?xf32> to tensor<3x3x1x1xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG4]][0, 0, %[[ARG1]], %[[ARG3]]]
// CHECK-SAME:         [8, 8, 1, 1] [1, 1, 1, 1] : tensor<8x8x?x?xf32> to tensor<8x8x1x1xf32>
// CHECK:            %[[TF:.+]] = iree_linalg_ext.winograd.filter_transform
// CHECK-SAME:         output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
// CHECK-SAME:         ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE_0]]
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG4]]
// CHECK-SAME:         [0, 0, %[[ARG1]], %[[ARG3]]] [8, 8, 1, 1] [1, 1, 1, 1]
// CHECK-SAME:         tensor<8x8x1x1xf32> into tensor<8x8x?x?xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x?x?xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<8x8x?x?xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<8x8x?x?xf32>
// CHECK:      }

// -----

func.func @winograd_filter_transform_fchw(%arg0: tensor<128x64x3x3xf32>) -> tensor<8x8x64x128xf32> {
  %0 = tensor.empty() : tensor<8x8x64x128xf32>
  %1 = iree_linalg_ext.winograd.filter_transform
    output_tile_size(6) kernel_size(3) kernel_dimensions([2, 3])
    ins(%arg0 : tensor<128x64x3x3xf32>) outs(%0 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %1 : tensor<8x8x64x128xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.filter_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK:      func.func @winograd_filter_transform_fchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<128x64x3x3xf32>) ->
// CHECK-SAME:   tensor<8x8x64x128xf32> {
// CHECK-DAG:    %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x64x128xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C64]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x64x128xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x64x128xf32>) {
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[ARG1]], 0, 0]
// CHECK-SAME:        [1, 1, 3, 3] [1, 1, 1, 1] : tensor<128x64x3x3xf32> to tensor<1x1x3x3xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG4]][0, 0, %[[ARG1]], %[[ARG3]]]
// CHECK-SAME:         [8, 8, 1, 1] [1, 1, 1, 1] : tensor<8x8x64x128xf32> to tensor<8x8x1x1xf32>
// CHECK:            %[[TF:.+]] = iree_linalg_ext.winograd.filter_transform
// CHECK-SAME:         output_tile_size(6) kernel_size(3) kernel_dimensions([2, 3])
// CHECK-SAME:         ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE_0]]
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG4]]
// CHECK-SAME:         [0, 0, %[[ARG1]], %[[ARG3]]] [8, 8, 1, 1] [1, 1, 1, 1]
// CHECK-SAME:         tensor<8x8x1x1xf32> into tensor<8x8x64x128xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x64x128xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<8x8x64x128xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<8x8x64x128xf32>
// CHECK:      }

// -----

func.func @winograd_input_transform(%arg0: tensor<1x10x10x1280xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform
    output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<1x10x10x1280xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.input_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -6 + 10, 8)>
// CHECK:      func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:          %[[IMG_IDX0:.+]] = affine.apply #[[MAP]](%[[ARG1]])
// CHECK-DAG:          %[[IMG_SIZE0:.+]] = affine.min #[[MAP1]](%[[ARG1]])
// CHECK-DAG:          %[[IMG_IDX1:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK-DAG:          %[[IMG_SIZE1:.+]] = affine.min #[[MAP1]](%[[ARG3]])
// CHECK:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG5]]]
// CHECK-SAME:          [1, %[[IMG_SIZE0]], %[[IMG_SIZE1]], 1] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<1x?x?x1xf32>
// CHECK:              %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG6]][0, 0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]]]
// CHECK-SAME:           [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to tensor<8x8x1x1x1x1xf32>
// CHECK:              %[[TF:.+]] = iree_linalg_ext.winograd.input_transform
// CHECK-SAME:           output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-SAME:           ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_0]]
// CHECK:              %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG6]]
// CHECK-SAME:           [0, 0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK-SAME:           tensor<8x8x1x1x1x1xf32> into tensor<8x8x1x2x2x1280xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:            }
// CHECK:            scf.yield %[[RES2]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }

// -----

func.func @winograd_input_transform_memref(%arg0: memref<1x10x10x1280xf32>, %arg1: memref<8x8x1x2x2x1280xf32>) {
  iree_linalg_ext.winograd.input_transform
    output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : memref<1x10x10x1280xf32>) outs(%arg1 : memref<8x8x1x2x2x1280xf32>)
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.input_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -6 + 10, 8)>
// CHECK:      func.func @winograd_input_transform_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<1x10x10x1280xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:        scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:          scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:            scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C1]] {
// CHECK-DAG:          %[[IMG_IDX0:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK-DAG:          %[[IMG_SIZE0:.+]] = affine.min #[[MAP1]](%[[ARG2]])
// CHECK-DAG:          %[[IMG_IDX1:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK-DAG:          %[[IMG_SIZE1:.+]] = affine.min #[[MAP1]](%[[ARG3]])
// CHECK:              %[[SUBVIEW:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:           [0, %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG4]]] [1, %[[IMG_SIZE0]], %[[IMG_SIZE1]], 1] [1, 1, 1, 1]
// CHECK:              %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:           [0, 0, 0, %[[ARG2]], %[[ARG3]], %[[ARG4]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:              iree_linalg_ext.winograd.input_transform
// CHECK-SAME:           output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-SAME:           ins(%[[SUBVIEW]]
// CHECK-SAME:           outs(%[[SUBVIEW_0]]
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }

// -----

func.func @winograd_input_transform_dynamic(%arg0: tensor<2x34x34x128xf32>, %i0: index, %i1: index, %i2: index, %s0: index, %s1: index, %s2: index, %s3: index) -> tensor<8x8x?x?x?x?xf32> {
  %c64 = arith.constant 64 : index
  %c2 = arith.constant 2 : index
  %8 = affine.min affine_map<(d0) -> (d0 * -8 + 34, 16)>(%i0)
  %9 = affine.min affine_map<(d0) -> (d0 * -8 + 34, 24)>(%i1)
  %10 = affine.apply affine_map<(d0) -> (d0 * 8)>(%i0)
  %11 = affine.apply affine_map<(d0) -> (d0 * 8)>(%i1)
  %extracted_slice = tensor.extract_slice %arg0[0, %10, %11, %i2][%c2, %8, %9, %c64][1, 1, 1, 1] : tensor<2x34x34x128xf32> to tensor<?x?x?x?xf32>
  %13 = tensor.empty(%s0, %s1, %s2, %s3) : tensor<8x8x?x?x?x?xf32>
  %14 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<?x?x?x?xf32>) outs(%13 : tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32>
  return %14 : tensor<8x8x?x?x?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.input_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * -8 + 34, 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * -8 + 34, 24)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0)[s0] -> (d0 * -6 + s0, 8)>
// CHECK:      func.func @winograd_input_transform_dynamic(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x34x34x128xf32>
// CHECK-SAME:   %[[I0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[I1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[I2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S3:[a-zA-Z0-9_]+]]: index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[EXTRACT_S0:.+]] = affine.min #[[MAP]]()[%[[I0]]]
// CHECK-DAG:    %[[EXTRACT_S1:.+]] = affine.min #[[MAP1]]()[%[[I1]]]
// CHECK-DAG:    %[[EXTRACTED_INPUT:.+]] = tensor.extract_slice %[[ARG0]]{{.*}}[2, %[[EXTRACT_S0]], %[[EXTRACT_S1]], 64]
// CHECK-SAME:     tensor<2x34x34x128xf32> to tensor<2x?x?x64xf32>
// CHECK:        %[[D0:.+]] = tensor.empty(%[[S0]], %[[S1]], %[[S2]], %[[S3]]) : tensor<8x8x?x?x?x?xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S0]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x?x?x?x?xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S1]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x?x?x?x?xf32>) {
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S2]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<8x8x?x?x?x?xf32>) {
// CHECK:              %[[RES3:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S3]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x?x?x?xf32>) {
// CHECK-DAG:            %[[IMG_IDX0:.+]] = affine.apply #[[MAP3]](%[[ARG3]])
// CHECK-DAG:            %[[IMG_SIZE0:.+]] = affine.min #[[MAP4]](%[[ARG3]])
// CHECK-DAG:            %[[IMG_IDX1:.+]] = affine.apply #[[MAP3]](%[[ARG5]])
// CHECK-DAG:            %[[IMG_SIZE1:.+]] = affine.min #[[MAP4]](%[[ARG5]])
// CHECK:                %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[EXTRACTED_INPUT]][%[[ARG1]], %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG7]]]
// CHECK-SAME:            [1, %[[IMG_SIZE0]], %[[IMG_SIZE1]], 1] [1, 1, 1, 1] : tensor<2x?x?x64xf32> to tensor<1x?x?x1xf32>
// CHECK:                %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG8]][0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]], %[[ARG7]]]
// CHECK-SAME:             [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x?x?x?xf32> to tensor<8x8x1x1x1x1xf32>
// CHECK:                %[[TF:.+]] = iree_linalg_ext.winograd.input_transform
// CHECK-SAME:             output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-SAME:             ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:             outs(%[[EXTRACTED_SLICE_0]]
// CHECK:                %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG8]]
// CHECK-SAME:             [0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]], %[[ARG7]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK-SAME:             tensor<8x8x1x1x1x1xf32> into tensor<8x8x?x?x?x?xf32>
// CHECK:                scf.yield %[[INSERTED_SLICE]] : tensor<8x8x?x?x?x?xf32>
// CHECK:              }
// CHECK:              scf.yield %[[RES3]] : tensor<8x8x?x?x?x?xf32>
// CHECK:            }
// CHECK:            scf.yield %[[RES2]] : tensor<8x8x?x?x?x?xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<8x8x?x?x?x?xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<8x8x?x?x?x?xf32>
// CHECK:      }

// -----

func.func @winograd_input_transform_nchw(%arg0: tensor<1x1280x10x10xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform
    output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
    ins(%arg0 : tensor<1x1280x10x10xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.input_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -6 + 10, 8)>
// CHECK:      func.func @winograd_input_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1280x10x10xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:          %[[IMG_IDX0:.+]] = affine.apply #[[MAP]](%[[ARG1]])
// CHECK-DAG:          %[[IMG_SIZE0:.+]] = affine.min #[[MAP1]](%[[ARG1]])
// CHECK-DAG:          %[[IMG_IDX1:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK-DAG:          %[[IMG_SIZE1:.+]] = affine.min #[[MAP1]](%[[ARG3]])
// CHECK:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[ARG5]], %[[IMG_IDX0]], %[[IMG_IDX1]]]
// CHECK-SAME:          [1, 1, %[[IMG_SIZE0]], %[[IMG_SIZE1]]] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<1x1x?x?xf32>
// CHECK:              %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG6]][0, 0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]]]
// CHECK-SAME:           [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to tensor<8x8x1x1x1x1xf32>
// CHECK:              %[[TF:.+]] = iree_linalg_ext.winograd.input_transform
// CHECK-SAME:           output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// CHECK-SAME:           ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_0]]
// CHECK:              %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG6]]
// CHECK-SAME:           [0, 0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK-SAME:           tensor<8x8x1x1x1x1xf32> into tensor<8x8x1x2x2x1280xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:            }
// CHECK:            scf.yield %[[RES2]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }

// -----

func.func @winograd_output_transform(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x12x12x32xf32> {
  %0 = tensor.empty() : tensor<1x12x12x32xf32>
  %1 = iree_linalg_ext.winograd.output_transform
        output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
        ins(%arg0 : tensor<8x8x1x2x2x32xf32>) outs(%0 : tensor<1x12x12x32xf32>) -> tensor<1x12x12x32xf32>
  return %1 : tensor<1x12x12x32xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.output_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// CHECK-SAME:   tensor<1x12x12x32xf32> {
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<1x12x12x32xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<1x12x12x32xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x12x12x32xf32>) {
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<1x12x12x32xf32>) {
// CHECK-DAG:          %[[IMG_IDX0:.+]] = affine.apply #[[MAP]](%[[ARG1]])
// CHECK-DAG:          %[[IMG_IDX1:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK:              %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG6]][0, %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG5]]]
// CHECK-SAME:          [1, 6, 6, 1] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<1x6x6x1xf32>
// CHECK:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]]]
// CHECK-SAME:           [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x1x1x1x1xf32>
// CHECK:              %[[TF:.+]] = iree_linalg_ext.winograd.output_transform
// CHECK-SAME:           output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-SAME:           ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_0]]
// CHECK:              %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG6]]
// CHECK-SAME:           [0, %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG5]]] [1, 6, 6, 1] [1, 1, 1, 1]
// CHECK-SAME:           tensor<1x6x6x1xf32> into tensor<1x12x12x32xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE]] : tensor<1x12x12x32xf32>
// CHECK:            }
// CHECK:            scf.yield %[[RES2]] : tensor<1x12x12x32xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<1x12x12x32xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<1x12x12x32xf32>
// CHECK:      }

// -----

func.func @winograd_output_transform_memref(%arg0: memref<8x8x1x2x2x32xf32>, %arg1: memref<1x12x12x32xf32>) {
  iree_linalg_ext.winograd.output_transform
   output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
   ins(%arg0 : memref<8x8x1x2x2x32xf32>) outs(%arg1 : memref<1x12x12x32xf32>)
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.output_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -6 + 12, 6)>
// CHECK:      func.func @winograd_output_transform_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<8x8x1x2x2x32xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<1x12x12x32xf32>) {
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:          scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:            scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-DAG:          %[[IMG_IDX0:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK-DAG:          %[[IMG_SIZE0:.+]] = affine.min #[[MAP1]](%[[ARG2]])
// CHECK-DAG:          %[[IMG_IDX1:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK-DAG:          %[[IMG_SIZE1:.+]] = affine.min #[[MAP1]](%[[ARG3]])
// CHECK:              %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:           [0, %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG4]]] [1, %[[IMG_SIZE0]], %[[IMG_SIZE1]], 1] [1, 1, 1, 1]
// CHECK:              %[[SUBVIEW:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:           [0, 0, 0, %[[ARG2]], %[[ARG3]], %[[ARG4]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:              iree_linalg_ext.winograd.output_transform
// CHECK-SAME:           output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-SAME:           ins(%[[SUBVIEW]]
// CHECK-SAME:           outs(%[[SUBVIEW_0]]
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }

// -----

func.func @winograd_output_transform_dynamic(%arg0: tensor<8x8x?x?x?x?xf32>, %i0: index, %i1: index, %i2: index, %i3: index, %s0: index, %s1: index, %s2: index, %s3: index, %s4: index, %s5: index) -> tensor<?x?x?x?xf32> {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, %i0, %i1, %i2, %i3][8, 8, %s0, %s1, %s2, %s3][1, 1, 1, 1, 1, 1] : tensor<8x8x?x?x?x?xf32> to tensor<8x8x?x?x?x?xf32>
  %12 = tensor.empty(%s0, %s4, %s5, %s3) : tensor<?x?x?x?xf32>
  %13 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<8x8x?x?x?x?xf32>) outs(%12 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %13 : tensor<?x?x?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.output_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK:      func.func @winograd_output_transform_dynamic(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x?x?x?x?xf32>
// CHECK-SAME:   %[[I0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[I1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[I2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[I3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S5:[a-zA-Z0-9_]+]]: index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[EXTRACTED_INPUT:.+]] = tensor.extract_slice %[[ARG0]]{{.*}}[8, 8, %[[S0]], %[[S1]], %[[S2]], %[[S3]]]
// CHECK-SAME:     tensor<8x8x?x?x?x?xf32> to tensor<8x8x?x?x?x?xf32>
// CHECK:        %[[D0:.+]] = tensor.empty(%[[S0]], %[[S4]], %[[S5]], %[[S3]]) : tensor<?x?x?x?xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S0]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<?x?x?x?xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S1]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<?x?x?x?xf32>) {
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S2]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<?x?x?x?xf32>) {
// CHECK:              %[[RES3:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[S3]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<?x?x?x?xf32>) {
// CHECK-DAG:            %[[IMG_IDX0:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK-DAG:            %[[IMG_IDX1:.+]] = affine.apply #[[MAP]](%[[ARG5]])
// CHECK:                %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG8]][%[[ARG1]], %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG7]]]
// CHECK-SAME:            [1, 6, 6, 1] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x6x6x1xf32>
// CHECK:                %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[EXTRACTED_INPUT]][0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]], %[[ARG7]]]
// CHECK-SAME:             [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x?x?x?xf32> to tensor<8x8x1x1x1x1xf32>
// CHECK:                %[[TF:.+]] = iree_linalg_ext.winograd.output_transform
// CHECK-SAME:             output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// CHECK-SAME:             ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:             outs(%[[EXTRACTED_SLICE_0]]
// CHECK:                %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG8]]
// CHECK-SAME:             [%[[ARG1]], %[[IMG_IDX0]], %[[IMG_IDX1]], %[[ARG7]]] [1, 6, 6, 1]
// CHECK-SAME:             tensor<1x6x6x1xf32> into tensor<?x?x?x?xf32>
// CHECK:                scf.yield %[[INSERTED_SLICE]] : tensor<?x?x?x?xf32>
// CHECK:              }
// CHECK:              scf.yield %[[RES3]] : tensor<?x?x?x?xf32>
// CHECK:            }
// CHECK:            scf.yield %[[RES2]] : tensor<?x?x?x?xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<?x?x?x?xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<?x?x?x?xf32>
// CHECK:      }

// -----

func.func @winograd_output_transform_nchw(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32> {
  %0 = tensor.empty() : tensor<1x32x12x12xf32>
  %1 = iree_linalg_ext.winograd.output_transform
        output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
        ins(%arg0 : tensor<8x8x1x2x2x32xf32>) outs(%0 : tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
  return %1 : tensor<1x32x12x12xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.winograd.output_transform"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK:      func.func @winograd_output_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// CHECK-SAME:   tensor<1x32x12x12xf32> {
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<1x32x12x12xf32>
// CHECK:        %[[RES0:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:       iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<1x32x12x12xf32>) {
// CHECK:          %[[RES1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x32x12x12xf32>) {
// CHECK:            %[[RES2:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<1x32x12x12xf32>) {
// CHECK-DAG:          %[[IMG_IDX0:.+]] = affine.apply #[[MAP]](%[[ARG1]])
// CHECK-DAG:          %[[IMG_IDX1:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK:              %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG6]][0, %[[ARG5]], %[[IMG_IDX0]], %[[IMG_IDX1]]]
// CHECK-SAME:          [1, 1, 6, 6] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<1x1x6x6xf32>
// CHECK:              %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0, %[[ARG1]], %[[ARG3]], %[[ARG5]]]
// CHECK-SAME:           [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x1x1x1x1xf32>
// CHECK:              %[[TF:.+]] = iree_linalg_ext.winograd.output_transform
// CHECK-SAME:           output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// CHECK-SAME:           ins(%[[EXTRACTED_SLICE]]
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_0]]
// CHECK:              %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TF]] into %[[ARG6]]
// CHECK-SAME:           [0, %[[ARG5]], %[[IMG_IDX0]], %[[IMG_IDX1]]] [1, 1, 6, 6] [1, 1, 1, 1]
// CHECK-SAME:           tensor<1x1x6x6xf32> into tensor<1x32x12x12xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE]] : tensor<1x32x12x12xf32>
// CHECK:            }
// CHECK:            scf.yield %[[RES2]] : tensor<1x32x12x12xf32>
// CHECK:          }
// CHECK:          scf.yield %[[RES1]] : tensor<1x32x12x12xf32>
// CHECK:        }
// CHECK:        return %[[RES0]] : tensor<1x32x12x12xf32>
// CHECK:      }

// -----

func.func @attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32) outs(%0 : tensor<192x1024x64xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     } -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 30] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (-d0 + 192, 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 1024, 30)>
// CHECK-DAG:  #[[MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG:  #[[MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG:  #[[MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG:  #[[MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK-DAG:    %[[C30:.+]] = arith.constant 30 : index
// CHECK-DAG:    %[[C1_F32:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C192:.+]] = arith.constant 192 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C10]]
// CHECK-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<192x1024x64xf32>) {
// CHECK:          %[[D3:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C30]]
// CHECK-SAME:       iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:        %[[D2:.+]] = affine.min #[[MAP]](%[[ARG3]])
// CHECK-DAG:        %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG5]])
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[ARG5]], 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, 0] [%[[D2]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], 0, 0] [%[[D2]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG3]], %[[ARG5]], 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.attention
// CHECK-SAME:                     {indexing_maps = [#[[MAP_Q]], #[[MAP_K]], #[[MAP_V]], #[[MAP_S]], #[[MAP_O]]]}
// CHECK-SAME:                    ins(%[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_0]],
// CHECK-SAME:         %[[EXTRACTED_SLICE_1]], %[[C1_F32]] : tensor<?x?x64xf32>, tensor<?x1024x64xf32>, tensor<?x1024x64xf32>, f32)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE_2]] : tensor<?x?x64xf32>) {
// CHECK:                ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:                  iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK:                } -> tensor<?x?x64xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0]
// CHECK-SAME:         [%[[D2]], %[[D4]], 64] [1, 1, 1] : tensor<?x?x64xf32> into tensor<192x1024x64xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<192x1024x64xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D3]] : tensor<192x1024x64xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }

// -----

func.func @attention_float_mask(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>, %mask: tensor<192x1024x1024xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32, tensor<192x1024x1024xf32>) outs(%0 : tensor<192x1024x64xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     } -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 30] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (-d0 + 192, 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 1024, 30)>
// CHECK-DAG:  #[[MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG:  #[[MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG:  #[[MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG:  #[[MAP_M:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
// CHECK-DAG:  #[[MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK:      func.func @attention_float_mask(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG3:[a-zA-Z0-9_]+]]: tensor<192x1024x1024xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK-DAG:    %[[C30:.+]] = arith.constant 30 : index
// CHECK-DAG:    %[[C1_F32:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C192:.+]] = arith.constant 192 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C10]]
// CHECK-SAME:     iter_args(%[[ARG5:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<192x1024x64xf32>) {
// CHECK:          %[[D2:.+]] = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C30]]
// CHECK-SAME:       iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[ARG5]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG4]])
// CHECK-DAG:        %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG6]])
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG4]], %[[ARG6]], 0] [%[[D3]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0, 0] [%[[D3]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG4]], 0, 0] [%[[D3]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG4]], %[[ARG6]], 0] [%[[D3]],
// CHECK-SAME:         %[[D4]], 1024] [1, 1, 1] : tensor<192x1024x1024xf32> to tensor<?x?x1024xf32>
// CHECK:            %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG7]][%[[ARG4]], %[[ARG6]], 0] [%[[D3]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.attention
// CHECK-SAME:                     {indexing_maps = [#[[MAP_Q]], #[[MAP_K]], #[[MAP_V]], #[[MAP_S]], #[[MAP_M]], #[[MAP_O]]]}
// CHECK-SAME:                    ins(%[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_0]],
// CHECK-SAME:         %[[EXTRACTED_SLICE_1]], %[[C1_F32]], %[[EXTRACTED_SLICE_2]] : tensor<?x?x64xf32>, tensor<?x1024x64xf32>, tensor<?x1024x64xf32>, f32, tensor<?x?x1024xf32>)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE_3]] : tensor<?x?x64xf32>)
// CHECK:                ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:                  iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK:                } -> tensor<?x?x64xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG7]][%[[ARG4]], %[[ARG6]], 0]
// CHECK-SAME:         [%[[D3]], %[[D4]], 64] [1, 1, 1] : tensor<?x?x64xf32> into tensor<192x1024x64xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<192x1024x64xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D2]] : tensor<192x1024x64xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }

// -----

func.func @attention_bool_mask(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>, %mask: tensor<192x1024x1024xi1>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32, tensor<192x1024x1024xi1>) outs(%0 : tensor<192x1024x64xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     } -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 30] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (-d0 + 192, 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 1024, 30)>
// CHECK-DAG:  #[[MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG:  #[[MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG:  #[[MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG:  #[[MAP_M:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
// CHECK-DAG:  #[[MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK:      func.func @attention_bool_mask(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG3:[a-zA-Z0-9_]+]]: tensor<192x1024x1024xi1>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK-DAG:    %[[C30:.+]] = arith.constant 30 : index
// CHECK-DAG:    %[[C1_F32:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C192:.+]] = arith.constant 192 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C10]]
// CHECK-SAME:     iter_args(%[[ARG5:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<192x1024x64xf32>) {
// CHECK:          %[[D2:.+]] = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C30]]
// CHECK-SAME:       iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[ARG5]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG4]])
// CHECK-DAG:        %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG6]])
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG4]], %[[ARG6]], 0] [%[[D3]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0, 0] [%[[D3]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG4]], 0, 0] [%[[D3]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG4]], %[[ARG6]], 0] [%[[D3]],
// CHECK-SAME:         %[[D4]], 1024] [1, 1, 1] : tensor<192x1024x1024xi1> to tensor<?x?x1024xi1>
// CHECK:            %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG7]][%[[ARG4]], %[[ARG6]], 0] [%[[D3]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[D5:.+]] = iree_linalg_ext.attention
// CHECK-SAME:                     {indexing_maps = [#[[MAP_Q]], #[[MAP_K]], #[[MAP_V]], #[[MAP_S]], #[[MAP_M]], #[[MAP_O]]]}
// CHECK-SAME:                    ins(%[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_0]],
// CHECK-SAME:         %[[EXTRACTED_SLICE_1]], %[[C1_F32]], %[[EXTRACTED_SLICE_2]] : tensor<?x?x64xf32>, tensor<?x1024x64xf32>, tensor<?x1024x64xf32>, f32, tensor<?x?x1024xi1>)
// CHECK-SAME:         outs(%[[EXTRACTED_SLICE_3]] : tensor<?x?x64xf32>)
// CHECK:                ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:                  iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK:              } -> tensor<?x?x64xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG7]][%[[ARG4]], %[[ARG6]], 0]
// CHECK-SAME:         [%[[D3]], %[[D4]], 64] [1, 1, 1] : tensor<?x?x64xf32> into tensor<192x1024x64xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<192x1024x64xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D2]] : tensor<192x1024x64xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }

// -----

func.func @attention_memref(%query: memref<192x1024x64xf32>, %key: memref<192x1024x64xf32>, %value: memref<192x1024x64xf32>, %output: memref<192x1024x64xf32>) {
  %scale = arith.constant 1.0 : f32
  iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : memref<192x1024x64xf32>, memref<192x1024x64xf32>, memref<192x1024x64xf32>, f32) outs(%output : memref<192x1024x64xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 30] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (-d0 + 192, 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 1024, 30)>
// CHECK-DAG:  #[[MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG:  #[[MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG:  #[[MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG:  #[[MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK:      func.func @attention_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   memref<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: memref<192x1024x64xf32>, %[[ARG3:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   memref<192x1024x64xf32>) {
// CHECK-DAG:    %[[C1_F32:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:        %[[C30:.+]] = arith.constant 30 : index
// CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:        %[[C192:.+]] = arith.constant 192 : index
// CHECK-DAG:        %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:        %[[C10:.+]] = arith.constant 10 : index
// CHECK:        scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C10]] {
// CHECK:          scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C30]] {
// CHECK-DAG:        %[[D0:.+]] = affine.min #[[MAP]](%[[ARG4]])
// CHECK-DAG:        %[[D1:.+]] = affine.min #[[MAP1]](%[[ARG5]])
// CHECK:            %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[ARG4]], %[[ARG5]], 0] [%[[D0]], %[[D1]], 64] [1, 1,
// CHECK-SAME:         1] : memref<192x1024x64xf32> to memref<?x?x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]][%[[ARG4]], 0, 0] [%[[D0]], 1024, 64] [1, 1, 1] :
// CHECK-SAME:         memref<192x1024x64xf32> to memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_1:.+]] = memref.subview %[[ARG2]][%[[ARG4]], 0, 0] [%[[D0]], 1024, 64] [1, 1, 1] :
// CHECK-SAME:         memref<192x1024x64xf32> to memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_2:.+]] = memref.subview %[[ARG3]][%[[ARG4]], %[[ARG5]], 0] [%[[D0]], %[[D1]], 64] [1, 1,
// CHECK-SAME:         1] : memref<192x1024x64xf32> to memref<?x?x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:            iree_linalg_ext.attention
// CHECK-SAME:                     {indexing_maps = [#[[MAP_Q]], #[[MAP_K]], #[[MAP_V]], #[[MAP_S]], #[[MAP_O]]]}
// CHECK-SAME:         ins(%[[SUBVIEW]], %[[SUBVIEW_0]], %[[SUBVIEW_1]], %[[C1_F32]] : memref<?x?x64xf32,
// CHECK-SAME:         strided<[65536, 64, 1], offset: ?>>, memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>,
// CHECK-SAME:         memref<?x1024x64xf32, strided<[65536, 64, 1], offset: ?>>, f32) outs(%[[SUBVIEW_2]] :
// CHECK-SAME:         memref<?x?x64xf32, strided<[65536, 64, 1], offset: ?>>)
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }

// -----

func.func @attention_fusion(
    %query: tensor<2x10x4096x64xf16>,
    %key: tensor<2x10x4096x64xf16>,
    %value: tensor<2x10x4096x64xf16>,
    %scale : f16, %bias : tensor<10x64xf16>) -> tensor<2x10x4096x64xf16> {
  %0 = tensor.empty() : tensor<2x10x4096x64xf16>
  %1 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]}
      ins(%query, %key, %value, %scale : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, f16)
      outs(%0 : tensor<2x10x4096x64xf16>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     } -> tensor<2x10x4096x64xf16>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%1, %bias : tensor<2x10x4096x64xf16>, tensor<10x64xf16>)
      outs(%0 : tensor<2x10x4096x64xf16>) {
    ^bb0(%b0 : f16, %b1 : f16, %b2 : f16):
      %3 = arith.addf %b0, %b1 : f16
      linalg.yield %3 : f16
  } -> tensor<2x10x4096x64xf16>
  return %2 : tensor<2x10x4096x64xf16>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.generic"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %2, %loops = transform.structured.tile_using_forall %1 tile_sizes [1, 1, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %_, %__ = transform.structured.fuse_into_containing_op %0 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @attention_fusion(
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x10x4096x64xf16>
//       CHECK:   %[[RESULT:.+]] = scf.forall
//  CHECK-SAME:       shared_outs(%[[OUTS:.+]] = %[[EMPTY]])
//       CHECK:     %[[OUTS_SLICE:.+]] = tensor.extract_slice %[[OUTS]]
//       CHECK:     %[[ATTENTION_SLICE:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:         outs(%[[OUTS_SLICE]] :
// CHECK:                ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:                  iree_linalg_ext.yield %[[SCORE]] : f32
//       CHECK:     %[[BIAS_SLICE:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ATTENTION_SLICE]],
//  CHECK-SAME:         outs(%[[OUTS_SLICE]] :
//       CHECK:     tensor.parallel_insert_slice %[[BIAS_SLICE]] into %[[OUTS]]
//       CHECK:   return %[[RESULT]]

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

// CHECK-DAG: #[[$IDXMAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-DAG: #[[$IDXMAP1:.+]] = affine_map<(d0) -> (d0 * 128)>
// CHECK-DAG: #[[$IDXMAP2:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
// CHECK-LABEL: @online_attention
// CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]], %[[IV2:.+]]) in (48, 8, 2)
// CHECK-DAG:   %[[I0:.+]] = affine.apply #[[$IDXMAP0]](%[[IV0]])
// CHECK-DAG:   %[[I1:.+]] = affine.apply #[[$IDXMAP1]](%[[IV1]])
// CHECK-DAG:   %[[I2:.+]] = affine.apply #[[$IDXMAP2]](%[[IV2]])
// CHECK-DAG:  %[[Q:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], 0] [4, 128, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x128x64xf32>
// CHECK-DAG:  %[[K:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], 0, 0] [4, 1024, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x1024x64xf32>
// CHECK-DAG:  %[[V:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], 0, %[[I2]]] [4, 1024, 32] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x1024x32xf32>
// CHECK-DAG:  %[[O:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], %[[I2]]] [4, 128, 32] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x128x32xf32>
// CHECK-DAG:  %[[M:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]]] [4, 128] [1, 1] : tensor<192x1024xf32> to tensor<4x128xf32>
// CHECK-DAG:  %[[S:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]]] [4, 128] [1, 1] : tensor<192x1024xf32> to tensor<4x128xf32>
// CHECK-DAG: iree_linalg_ext.online_attention
// CHECK-SAME: {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]], #[[$MAP5]], #[[$MAP5]]]}
// CHECK-SAME: ins(%[[Q]], %[[K]], %[[V]], %{{.*}} : tensor<4x128x64xf32>, tensor<4x1024x64xf32>, tensor<4x1024x32xf32>, f32)
// CHECK-SAME: outs(%[[O]], %[[M]], %[[S]] : tensor<4x128x32xf32>, tensor<4x128xf32>, tensor<4x128xf32>)
// CHECK: scf.forall.in_parallel

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128, 0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}


// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_partial_reduction(%query: tensor<192x?x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>) -> (tensor<192x?x64xf32>, tensor<192x?xf32>) {
  %scale = arith.constant 1.0 : f32

  %c1 = arith.constant 1 : index

  %m = tensor.dim %query, %c1 : tensor<192x?x64xf32>
  %k2 = tensor.dim %key, %c1 : tensor<192x?x64xf32>

  %output_empty = tensor.empty(%m) : tensor<192x?x64xf32>
  %row_red_empty = tensor.empty(%m) : tensor<192x?xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x?x64xf32>) -> tensor<192x?x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x?xf32>) -> tensor<192x?xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x?xf32>) -> tensor<192x?xf32>

  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale : tensor<192x?x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x?x64xf32>, tensor<192x?xf32>, tensor<192x?xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x?x64xf32>, tensor<192x?xf32>, tensor<192x?xf32>

  return %out#0, %out#2 : tensor<192x?x64xf32>, tensor<192x?xf32>
}

// CHECK-LABEL: func.func @online_attention_partial_reduction
// CHECK-SAME: (%[[Q:.+]]: tensor<192x?x64xf32>, %[[K:.+]]: tensor<192x?x64xf32>, %[[V:.+]]: tensor<192x?x64xf32>)

// CHECK-DAG:  %[[M:.+]] = tensor.dim %[[Q]], %c1 : tensor<192x?x64xf32>
// CHECK-DAG:  %[[K2:.+]] = tensor.dim %[[K]], %c1 : tensor<192x?x64xf32>

// CHECK-DAG:  %[[OUT_E:.+]] = tensor.empty(%[[M]]) : tensor<192x?x64xf32>
// CHECK-DAG:  %[[RED_E:.+]] = tensor.empty(%[[M]]) : tensor<192x?xf32>

// CHECK-DAG:  %[[SUM_INIT:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[MAX_INIT:.+]] = arith.constant -3.40282347E+38 : f32

// CHECK-DAG:  %[[OUT:.+]] = linalg.fill ins(%[[SUM_INIT]] : f32) outs(%[[OUT_E]] : tensor<192x?x64xf32>) -> tensor<192x?x64xf32>
// CHECK-DAG:  %[[MAX:.+]] = linalg.fill ins(%[[MAX_INIT]] : f32) outs(%[[RED_E]] : tensor<192x?xf32>) -> tensor<192x?xf32>
// CHECK-DAG:  %[[SUM:.+]] = linalg.fill ins(%[[SUM_INIT]] : f32) outs(%[[RED_E]] : tensor<192x?xf32>) -> tensor<192x?xf32>

// CHECK-DAG:  %[[OUT_PART_E:.+]] = tensor.empty(%[[M]]) : tensor<192x?x64x32xf32>
// CHECK-DAG:  %[[RED_PART_E:.+]] = tensor.empty(%[[M]]) : tensor<192x?x32xf32>

// CHECK-DAG:  %[[OUT_PART:.+]] = linalg.fill ins(%[[SUM_INIT]] : f32) outs(%[[OUT_PART_E]] : tensor<192x?x64x32xf32>) -> tensor<192x?x64x32xf32>
// CHECK-DAG:  %[[MAX_PART:.+]] = linalg.fill ins(%[[MAX_INIT]] : f32) outs(%[[RED_PART_E]] : tensor<192x?x32xf32>) -> tensor<192x?x32xf32>
// CHECK-DAG:  %[[SUM_PART:.+]] = linalg.fill ins(%[[SUM_INIT]] : f32) outs(%[[RED_PART_E]] : tensor<192x?x32xf32>) -> tensor<192x?x32xf32>

// CHECK:      %[[ITER:.+]]:3 = scf.for %[[IV:.+]] = %c0 to %[[K2]] step %c32
// CHECK-SAME: iter_args(%[[OUT_ITER:.+]] = %[[OUT_PART]], %[[MAX_ITER:.+]] = %[[MAX_PART]], %[[SUM_ITER:.+]] = %[[SUM_PART]])
// CHECK:         %[[MIN:.+]] = affine.min
// CHECK:         %[[Q_SLICE:.+]] = tensor.extract_slice %[[Q]][0, 0, 0] [192, %[[M]], 64] [1, 1, 1] : tensor<192x?x64xf32> to tensor<192x?x64xf32>
// CHECK:         %[[K_SLICE:.+]] = tensor.extract_slice %[[K]][0, %[[IV]], 0] [192, %[[MIN]], 64] [1, 1, 1] : tensor<192x?x64xf32> to tensor<192x?x64xf32>
// CHECK:         %[[V_SLICE:.+]] = tensor.extract_slice %[[V]][0, %[[IV]], 0] [192, %[[MIN]], 64] [1, 1, 1] : tensor<192x?x64xf32> to tensor<192x?x64xf32>

// CHECK:         %[[OUT_SLICE:.+]] = tensor.extract_slice %[[OUT_ITER]][0, 0, 0, 0] [192, %[[M]], 64, %[[MIN]]] [1, 1, 1, 1] : tensor<192x?x64x32xf32> to tensor<192x?x64x?xf32>
// CHECK:         %[[MAX_SLICE:.+]] = tensor.extract_slice %[[MAX_ITER]][0, 0, 0] [192, %[[M]], %[[MIN]]] [1, 1, 1] : tensor<192x?x32xf32> to tensor<192x?x?xf32>
// CHECK:         %[[SUM_SLICE:.+]] = tensor.extract_slice %[[SUM_ITER]][0, 0, 0] [192, %[[M]], %[[MIN]]] [1, 1, 1] : tensor<192x?x32xf32> to tensor<192x?x?xf32>

// CHECK:         %[[OATT:.+]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:    ins(%[[Q_SLICE]], %[[K_SLICE]], %[[V_SLICE]]
// CHECK-SAME:    outs(%[[OUT_SLICE]], %[[MAX_SLICE]], %[[SUM_SLICE]]

// CHECK:         %[[OUT_NEXT:.+]] = tensor.insert_slice %[[OATT]]#0 into %[[OUT_ITER]][0, 0, 0, 0] [192, %[[M]], 64, %[[MIN]]] [1, 1, 1, 1] : tensor<192x?x64x?xf32> into tensor<192x?x64x32xf32>
// CHECK:         %[[MAX_NEXT:.+]] = tensor.insert_slice %[[OATT]]#1 into %[[MAX_ITER]][0, 0, 0] [192, %[[M]], %[[MIN]]] [1, 1, 1] : tensor<192x?x?xf32> into tensor<192x?x32xf32>
// CHECK:         %[[SUM_NEXT:.+]] = tensor.insert_slice %[[OATT]]#2 into %[[SUM_ITER]][0, 0, 0] [192, %[[M]], %[[MIN]]] [1, 1, 1] : tensor<192x?x?xf32> into tensor<192x?x32xf32>

// CHECK:         scf.yield %[[OUT_NEXT]], %[[MAX_NEXT]], %[[SUM_NEXT]]

// CHECK: %[[MAX_RED:.+]] = linalg.reduce ins(%[[ITER]]#1
// CHECK-SAME: dimensions = [2]
// CHECK:     arith.maximumf
// CHECK:     linalg.yield

// CHECK: %[[NORM:.+]] = linalg.generic
// CHECK:     arith.subf
// CHECK:     math.exp2
// CHECK:     linalg.yield

// CHECK: %[[NORM_SUM:.+]] = linalg.generic
// CHECK-SAME: ins(%[[NORM]]
// CHECK-SAME: outs(%[[ITER]]#2
// CHECK:     arith.mulf
// CHECK:     linalg.yield

// CHECK: %[[SUM_RED:.+]] = linalg.reduce ins(%[[NORM_SUM]]
// CHECK-SAME: dimensions = [2]
// CHECK:      arith.addf
// CHECK:      linalg.yield

// CHECK: %[[NORM_ACC:.+]] = linalg.generic
// CHECK-SAME: ins(%[[NORM]]
// CHECK-SAME: outs(%[[ITER]]#0
// CHECK:     arith.mulf
// CHECK:     linalg.yield

// CHECK: %[[ACC_RED:.+]] = linalg.reduce ins(%[[NORM_ACC]]
// CHECK-SAME: dimensions = [3]
// CHECK:      arith.addf
// CHECK:      linalg.yield

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %fill_op:3, %split, %merge:3, %forop = transform.structured.tile_reduction_using_for %0 by tile_sizes = [0, 0, 0, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_float_mask(%query: tensor<192x1024x64xf32>,
                            %key: tensor<192x1024x64xf32>,
                            %value: tensor<192x1024x64xf32>,
                            %mask: tensor<192x1024x1024xf32>)
                            -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  // Adjust the operation to correctly handle the mask
  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32, tensor<192x1024x1024xf32>)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

// CHECK-DAG: #[[$IDXMAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-DAG: #[[$IDXMAP1:.+]] = affine_map<(d0) -> (d0 * 128)>
// CHECK-DAG: #[[$IDXMAP2:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
// CHECK-LABEL: @online_attention_float_mask
// CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]], %[[IV2:.+]]) in (48, 8, 2)
// CHECK-DAG:   %[[I0:.+]] = affine.apply #[[$IDXMAP0]](%[[IV0]])
// CHECK-DAG:   %[[I1:.+]] = affine.apply #[[$IDXMAP1]](%[[IV1]])
// CHECK-DAG:   %[[I2:.+]] = affine.apply #[[$IDXMAP2]](%[[IV2]])
// CHECK-DAG:  %[[Q:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], 0] [4, 128, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x128x64xf32>
// CHECK-DAG:  %[[K:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], 0, 0] [4, 1024, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x1024x64xf32>
// CHECK-DAG:  %[[V:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], 0, %[[I2]]] [4, 1024, 32] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x1024x32xf32>
// CHECK-DAG:  %[[MASK:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], 0] [4, 128, 1024] [1, 1, 1] : tensor<192x1024x1024xf32> to tensor<4x128x1024xf32>
// CHECK-DAG:  %[[O:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], %[[I2]]] [4, 128, 32] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x128x32xf32>
// CHECK-DAG:  %[[M:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]]] [4, 128] [1, 1] : tensor<192x1024xf32> to tensor<4x128xf32>
// CHECK-DAG:  %[[S:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]]] [4, 128] [1, 1] : tensor<192x1024xf32> to tensor<4x128xf32>
// CHECK-DAG: iree_linalg_ext.online_attention
// CHECK-SAME: {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]], #[[$MAP5]], #[[$MAP6]], #[[$MAP6]]]}
// CHECK-SAME: ins(%[[Q]], %[[K]], %[[V]], %{{.*}}, %[[MASK]] : tensor<4x128x64xf32>, tensor<4x1024x64xf32>, tensor<4x1024x32xf32>, f32, tensor<4x128x1024xf32>)
// CHECK-SAME: outs(%[[O]], %[[M]], %[[S]] : tensor<4x128x32xf32>, tensor<4x128xf32>, tensor<4x128xf32>)
// CHECK: scf.forall.in_parallel

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128, 0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_bool_mask(%query: tensor<192x1024x64xf32>,
                            %key: tensor<192x1024x64xf32>,
                            %value: tensor<192x1024x64xf32>,
                            %mask: tensor<192x1024x1024xi1>)
                            -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  // Adjust the operation to correctly handle the mask
  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32, tensor<192x1024x1024xi1>)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}


// CHECK-DAG: #[[$IDXMAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-DAG: #[[$IDXMAP1:.+]] = affine_map<(d0) -> (d0 * 128)>
// CHECK-DAG: #[[$IDXMAP2:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
// CHECK-LABEL: @online_attention_bool_mask
// CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]], %[[IV2:.+]]) in (48, 8, 2)
// CHECK-DAG:   %[[I0:.+]] = affine.apply #[[$IDXMAP0]](%[[IV0]])
// CHECK-DAG:   %[[I1:.+]] = affine.apply #[[$IDXMAP1]](%[[IV1]])
// CHECK-DAG:   %[[I2:.+]] = affine.apply #[[$IDXMAP2]](%[[IV2]])
// CHECK-DAG:  %[[Q:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], 0] [4, 128, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x128x64xf32>
// CHECK-DAG:  %[[K:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], 0, 0] [4, 1024, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x1024x64xf32>
// CHECK-DAG:  %[[V:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], 0, %[[I2]]] [4, 1024, 32] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x1024x32xf32>
// CHECK-DAG:  %[[MASK:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], 0] [4, 128, 1024] [1, 1, 1] : tensor<192x1024x1024xi1> to tensor<4x128x1024xi1>
// CHECK-DAG:  %[[O:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]], %[[I2]]] [4, 128, 32] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<4x128x32xf32>
// CHECK-DAG:  %[[M:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]]] [4, 128] [1, 1] : tensor<192x1024xf32> to tensor<4x128xf32>
// CHECK-DAG:  %[[S:.+]] = tensor.extract_slice %{{.*}}[%[[I0]], %[[I1]]] [4, 128] [1, 1] : tensor<192x1024xf32> to tensor<4x128xf32>
// CHECK-DAG: iree_linalg_ext.online_attention
// CHECK-SAME: {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]], #[[$MAP5]], #[[$MAP6]], #[[$MAP6]]]}
// CHECK-SAME: ins(%[[Q]], %[[K]], %[[V]], %{{.*}}, %[[MASK]] : tensor<4x128x64xf32>, tensor<4x1024x64xf32>, tensor<4x1024x32xf32>, f32, tensor<4x128x1024xi1>)
// CHECK-SAME: outs(%[[O]], %[[M]], %[[S]] : tensor<4x128x32xf32>, tensor<4x128xf32>, tensor<4x128xf32>)
// CHECK: scf.forall.in_parallel

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128, 0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @tile_custom_op_simple(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?x?xf32>, %b2 : tensor<?x?xf32>) :
      %1 = linalg.matmul ins(%b0, %b1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%b2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      iree_linalg_ext.yield %1 : tensor<?x?xf32>
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.custom_op"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 4)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0) -> (d0 * 128)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0)[s0] -> (d0 * -4 + s0, 4)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0)[s0] -> (d0 * -128 + s0, 128)>
//      CHECK: func @tile_custom_op_simple(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[UB0:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[UB1:.+]] = affine.apply #[[MAP1]]()[%[[N]]]
//      CHECK:   %[[RETURN:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) in (%[[UB0]], %[[UB1]])
// CHECK-SAME:       shared_outs(%[[INIT:.+]] = %[[ARG2]])
//  CHECK-DAG:     %[[OFFSET_M:.+]] = affine.apply #[[MAP2]](%[[IV0]])
//  CHECK-DAG:     %[[OFFSET_N:.+]] = affine.apply #[[MAP3]](%[[IV1]])
//  CHECK-DAG:     %[[TILESIZE_M:.+]] = affine.min #[[MAP4]](%[[IV0]])[%[[M]]]
//  CHECK-DAG:     %[[TILESIZE_N:.+]] = affine.min #[[MAP5]](%[[IV1]])[%[[N]]]
//  CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[OFFSET_M]], 0] [%[[TILESIZE_M]], %[[K]]]
//  CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[OFFSET_N]]] [%[[K]], %[[TILESIZE_N]]]
//  CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[OFFSET_M]], %[[OFFSET_N]]] [%[[TILESIZE_M]], %[[TILESIZE_N]]]
//      CHECK:     %[[CUSTOM_OP:.+]] = iree_linalg_ext.custom_op
// CHECK-SAME:         ins(%[[LHS_SLICE]], %[[RHS_SLICE]] :
// CHECK-SAME:         outs(%[[INIT_SLICE]] :
//      CHECK:     scf.forall.in_parallel
//      CHECK:       tensor.parallel_insert_slice %[[CUSTOM_OP]] into %[[INIT]][%[[OFFSET_M]], %[[OFFSET_N]]] [%[[TILESIZE_M]], %[[TILESIZE_N]]]
//      CHECK:   return %[[RETURN]]

// -----

func.func @tile_custom_op_with_symbolic_indexing_maps(%lhs0 : tensor<?x?xf32>,
    %rhs0: tensor<?x?xf32>, %init0 : tensor<?x?xf32>,
    %rhs1 : tensor<?x?xf32>, %init1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (d0, s0)>,
                       affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                                   affine_map<(d0, d1)[s0, s1] -> (d0, s1)>,
                       affine_map<(d0, d1)[s0, s1] -> (s1, d1)>,
                               affine_map<(d0, d1)[s0, s1] -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>]}
      ins(%lhs0, %rhs0, %init0, %rhs1 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init1 : tensor<?x?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?x?xf32>, %b2 : tensor<?x?xf32>, %b3 : tensor<?x?xf32>, %b4 : tensor<?x?xf32>) :
      %1 = linalg.matmul ins(%b0, %b1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%b2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %2 = linalg.matmul ins(%1, %b3 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%b4 : tensor<?x?xf32>) -> tensor<?x?xf32>
      iree_linalg_ext.yield %2 : tensor<?x?xf32>
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.custom_op"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 4)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0) -> (d0 * 128)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0)[s0] -> (d0 * -4 + s0, 4)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0)[s0] -> (d0 * -128 + s0, 128)>
//      CHECK: func @tile_custom_op_with_symbolic_indexing_maps(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[M0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N1:.+]] = tensor.dim %[[ARG3]], %[[C1]]
//  CHECK-DAG:   %[[UB0:.+]] = affine.apply #[[MAP0]]()[%[[M0]]]
//  CHECK-DAG:   %[[UB1:.+]] = affine.apply #[[MAP1]]()[%[[N1]]]
//      CHECK:   %[[RETURN:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) in (%[[UB0]], %[[UB1]])
// CHECK-SAME:       shared_outs(%[[INIT:.+]] = %[[ARG4]])
//  CHECK-DAG:     %[[OFFSET_M:.+]] = affine.apply #[[MAP2]](%[[IV0]])
//  CHECK-DAG:     %[[OFFSET_N:.+]] = affine.apply #[[MAP3]](%[[IV1]])
//  CHECK-DAG:     %[[TILESIZE_M:.+]] = affine.min #[[MAP4]](%[[IV0]])[%[[M0]]]
//  CHECK-DAG:     %[[TILESIZE_N:.+]] = affine.min #[[MAP5]](%[[IV1]])[%[[N1]]]
//  CHECK-DAG:     %[[K0:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:     %[[K0_1:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:     %[[N0:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:     %[[N0_1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//  CHECK-DAG:     %[[K1:.+]] = tensor.dim %[[ARG3]], %[[C0]]
//  CHECK-DAG:     %[[ARG0_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[OFFSET_M]], 0] [%[[TILESIZE_M]], %[[K0]]]
//  CHECK-DAG:     %[[ARG1_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, 0] [%[[K0_1]], %[[N0]]]
//  CHECK-DAG:     %[[ARG2_SLICE:.+]] = tensor.extract_slice %[[ARG2]][%[[OFFSET_M]], 0] [%[[TILESIZE_M]], %[[N0_1]]]
//  CHECK-DAG:     %[[ARG3_SLICE:.+]] = tensor.extract_slice %[[ARG3]][0, %[[OFFSET_N]]] [%[[K1]], %[[TILESIZE_N]]]
//  CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[OFFSET_M]], %[[OFFSET_N]]] [%[[TILESIZE_M]], %[[TILESIZE_N]]]
//      CHECK:     %[[TILED_OP:.+]] = iree_linalg_ext.custom_op
// CHECK-SAME:         ins(%[[ARG0_SLICE]], %[[ARG1_SLICE]], %[[ARG2_SLICE]], %[[ARG3_SLICE]] :
// CHECK-SAME:         outs(%[[INIT_SLICE]] :
//      CHECK:     scf.forall.in_parallel
//      CHECK:       tensor.parallel_insert_slice %[[TILED_OP]] into %[[INIT]][%[[OFFSET_M]], %[[OFFSET_N]]] [%[[TILESIZE_M]], %[[TILESIZE_N]]]
//      CHECK:   return %[[RETURN]]

// -----

func.func @custom_op_do_not_tile_empty_map(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<() -> ()>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?x?xf32>, %b2 : tensor<?x?xf32>):
      %1 = linalg.matmul ins(%b0, %b1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%b2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      iree_linalg_ext.yield %1 : tensor<?x?xf32>
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.custom_op"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @custom_op_do_not_tile_empty_map(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
//       CHECK:   scf.forall
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//       CHECK:     iree_linalg_ext.custom_op
//  CHECK-SAME:         ins(%[[SLICE]], %[[ARG1]]

// -----

func.func @custom_op_tile_scalar_args(%arg0 : tensor<?x?xf32>,
    %arg1 : f32, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> ()>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, f32) outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : f32, %b2 : tensor<?x?xf32>):
      %1 = linalg.generic {
          iterator_types = ["parallel", "parallel", "reduction"],
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> ()>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>]}
          ins(%b0, %b1 : tensor<?x?xf32>, f32)
          outs(%b2 : tensor<?x?xf32>) {
        ^bb0(%bb0: f32, %bb1: f32, %bb2: f32):
          %2 = arith.mulf %bb0, %bb1 : f32
          %3 = arith.addf %2, %bb2 : f32
          linalg.yield %3 : f32
      } -> tensor<?x?xf32>
      iree_linalg_ext.yield %1 : tensor<?x?xf32>
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.custom_op"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @custom_op_tile_scalar_args(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>, %[[ARG1:[a-zA-Z0-9]+]]: f32,
//       CHECK:   scf.forall
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//       CHECK:     iree_linalg_ext.custom_op
//  CHECK-SAME:         ins(%[[SLICE]], %[[ARG1]]

// -----

func.func @custom_op_index_handling(%arg0 : tensor<?x?xindex>,
    %arg2 : tensor<?x?xindex>) -> tensor<?x?xindex> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<reduction>]}
      ins(%arg0 : tensor<?x?xindex>) outs(%arg2 : tensor<?x?xindex>) {
    ^bb0(%b0 : tensor<?x?xindex>, %b2 : tensor<?x?xindex>):
      %1 = iree_linalg_ext.index 0 : index
      %2 = linalg.generic {
          iterator_types = ["parallel", "parallel", "reduction"],
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> ()>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>]}
          ins(%b0, %1 : tensor<?x?xindex>, index)
          outs(%b2 : tensor<?x?xindex>) {
        ^bb0(%bb0: index, %bb1: index, %bb2: index):
          %2 = arith.muli %bb0, %bb1 : index
          %3 = arith.addi %2, %bb2 : index
          linalg.yield %3 : index
      } -> tensor<?x?xindex>
      iree_linalg_ext.yield %2 : tensor<?x?xindex>
  } -> tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.custom_op"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %grid = transform.structured.tile_using_forall %0 tile_sizes [4, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: #[[MAP:.+]] = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
//      CHECK: func @custom_op_index_handling(%[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xindex>,
//      CHECK:   scf.forall (%[[IV:[a-zA-Z0-9]+]],
//      CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//      CHECK:     iree_linalg_ext.custom_op
// CHECK-SAME:         ins(%[[SLICE]]
//      CHECK:       %[[NEW_INDEX:.+]] = iree_linalg_ext.index 0 : index
//      CHECK:       %[[INDEX:.+]] = affine.apply #[[MAP]](%[[IV]])[%[[NEW_INDEX]]]
//      CHECK:       linalg.generic
// CHECK-SAME:           ins(%{{.+}}, %[[INDEX]] :

// -----

func.func @gather_1d_indices(%arg0 : memref<?x?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?x?xi32>) {
  iree_linalg_ext.gather
    dimension_map = [0]
    ins(%arg0, %arg1: memref<?x?xi32>, memref<?xi32>)
    outs(%arg2: memref<?x?xi32>)
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.gather"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [10, 20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
//      CHECK: func @gather_1d_indices
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[C20:.+]] = arith.constant 20
//  CHECK-DAG:   %[[C10:.+]] = arith.constant 10
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG2]], %[[C1]]
//      CHECK:   scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C10]]
//      CHECK:     scf.for %[[J:.+]] = %[[C0]] to %[[D1]] step %[[C20]]
//  CHECK-DAG:       %[[MIN0:.+]] = affine.min #[[MAP0]](%[[I]])[%[[D0]]]
//  CHECK-DAG:       %[[MIN1:.+]] = affine.min #[[MAP1]](%[[J]])[%[[D1]]]
//  CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[ARG2]][%[[I]], %[[J]]] [%[[MIN0]], %[[MIN1]]] [1, 1]
//  CHECK-DAG:       %[[INDEX:.+]] = memref.subview %[[ARG1]][%[[I]]] [%[[MIN0]]] [1]
//  CHECK-DAG:       %[[D2:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:       %[[SOURCE:.+]] = memref.subview %[[ARG0]][0, %[[J]]] [%[[D2]], %[[MIN1]]] [1, 1]
//      CHECK:       iree_linalg_ext.gather
// CHECK-SAME:           ins(%[[SOURCE]], %[[INDEX]]
// CHECK-SAME:           outs(%[[RESULT]]

// -----

func.func @gather_2d_indices(%arg0 : memref<?x?xi32>, %arg1 : memref<?x2xi32>, %arg2 : memref<?xi32>) {
  iree_linalg_ext.gather
    dimension_map = [0, 1]
    ins(%arg0, %arg1: memref<?x?xi32>, memref<?x2xi32>)
    outs(%arg2: memref<?xi32>)
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.gather"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [13] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 13)>
//      CHECK: func @gather_2d_indices
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[C13:.+]] = arith.constant 13
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG2]], %[[C0]]
//      CHECK:   scf.for %[[I:.+]] = %[[C0]] to %[[D0]] step %[[C13]]
//  CHECK-DAG:     %[[MIN:.+]] = affine.min #[[MAP0]](%[[I]])[%[[D0]]]
//  CHECK-DAG:     %[[RESULT:.+]] = memref.subview %[[ARG2]][%[[I]]] [%[[MIN]]] [1]
//  CHECK-DAG:     %[[INDEX:.+]] = memref.subview %[[ARG1]][%[[I]], 0] [%[[MIN]], 2] [1, 1]
//  CHECK-DAG:     %[[D1:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:     %[[D2:.+]] = memref.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:     %[[SOURCE:.+]] = memref.subview %[[ARG0]][0, 0] [%[[D1]], %[[D2]]] [1, 1]
//      CHECK:     iree_linalg_ext.gather
// CHECK-SAME:         ins(%[[SOURCE]], %[[INDEX]]
// CHECK-SAME:         outs(%[[RESULT]]

// -----

func.func @map_scatter_tensor(
    %input: tensor<?xf32>, %output: tensor<?xf32>
) -> tensor<?xf32> {
  %0 = iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %mask : index, i1
  } : tensor<?xf32> into tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.map_scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
//      CHECK: func @map_scatter_tensor
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[C8:.+]] = arith.constant 8
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[INPUT]], %[[C0]]
//  CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//      CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[C8]]
// CHECK-SAME:       iter_args(%[[LOOP_ARG:.+]] = %[[OUTPUT]])
//  CHECK-DAG:     %[[TILE_SIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[D0]]]
//  CHECK-DAG:     %[[INPUT_TILE:.+]] = tensor.extract_slice %[[INPUT]]
// CHECK-SAME:       [%[[IV]]] [%[[TILE_SIZE]]] [1]
//      CHECK:     iree_linalg_ext.map_scatter %[[INPUT_TILE]] into %[[LOOP_ARG]]
// CHECK-NEXT:       ^bb0(%[[IDX:.+]]: index):
//  CHECK-DAG:         %[[IDX_OFFSET:.+]] = affine.apply #[[MAP1]](%[[IDX]], %[[IV]])
//      CHECK:         iree_linalg_ext.yield %[[IDX_OFFSET]], %[[TRUE]]

// -----

func.func @map_scatter_memref(%input: memref<?xf32>, %output: memref<?xf32>) {
  iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %mask : index, i1
  } : memref<?xf32> into memref<?xf32>
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.map_scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
//      CHECK: func @map_scatter_memref
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[C8:.+]] = arith.constant 8
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[D0:.+]] = memref.dim %[[INPUT]], %[[C0]]
//  CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//      CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[D0]] step %[[C8]]
//  CHECK-DAG:     %[[TILE_SIZE:.+]] = affine.min #[[MAP]](%[[IV]])[%[[D0]]]
//  CHECK-DAG:     %[[INPUT_TILE:.+]] = memref.subview %[[INPUT]]
// CHECK-SAME:       [%[[IV]]] [%[[TILE_SIZE]]] [1]
//      CHECK:     iree_linalg_ext.map_scatter %[[INPUT_TILE]] into %[[OUTPUT]]
// CHECK-NEXT:       ^bb0(%[[IDX:.+]]: index):
//  CHECK-DAG:         %[[IDX_OFFSET:.+]] = affine.apply #[[MAP1]](%[[IDX]], %[[IV]])
//      CHECK:         iree_linalg_ext.yield %[[IDX_OFFSET]], %[[TRUE]]
