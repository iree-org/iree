// RUN: iree-opt --transform-dialect-interpreter --split-input-file -canonicalize -cse %s | FileCheck  %s

func.func @scatter_tiling_distribution(
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
transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %0 = transform.structured.match ops{["iree_linalg_ext.scatter"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %forall, %tiled_op = transform.structured.tile_using_forall %0 tile_sizes [10, 30, 0] { mapping = [#gpu.thread<y>, #gpu.thread<x>] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 10)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 30)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0) -> (d0 * 30)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0)[s0] -> (d0 * -30 + s0, 30)>
// CHECK:       func.func @scatter_tiling_distribution(
// CHECK-SAME:      %[[ORIGINAL:[a-zA-Z0-9_]+]]
// CHECK-SAME:      %[[INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:      %[[UPDATES:[a-zA-Z0-9_]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[D0:.+]] = tensor.dim %[[UPDATES]], %[[C0]]
// CHECK-DAG:     %[[D1:.+]] = tensor.dim %[[UPDATES]], %[[C1]]
// CHECK-DAG:     %[[UB0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
// CHECK-DAG:     %[[UB1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
// CHECK:         %[[RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[UB0]], %[[UB1]]) shared_outs(%[[ITER:.+]] = %[[ORIGINAL]])
// CHECK-DAG:       %[[I:.+]] = affine.apply #[[MAP2]](%[[IV0]])
// CHECK-DAG:       %[[I_SZ:.+]] = affine.min #[[MAP3]](%[[IV0]])[%[[D0]]]
// CHECK-DAG:       %[[J:.+]] = affine.apply #[[MAP4]](%[[IV1]])
// CHECK-DAG:       %[[J_SZ:.+]] = affine.min #[[MAP5]](%[[IV1]])[%[[D1]]]
// CHECK:           %[[UPDATES_TILE:.+]] = tensor.extract_slice %[[UPDATES]]
// CHECK-SAME:        [%[[I]], %[[J]]] [%[[I_SZ]], %[[J_SZ]]] [1, 1]
// CHECK:           %[[INDICES_TILE:.+]] = tensor.extract_slice %[[INDICES]]
// CHECK-SAME:        [%[[I]], 0] [%[[I_SZ]], 1] [1, 1]
// CHECK:           %[[ITER_D0:.+]] =  tensor.dim %[[ITER]], %[[C0]]
// CHECK:           %[[ITER_TILE:.+]] = tensor.extract_slice %[[ITER]]
// CHECK-SAME:        [0, %[[J]]] [%[[ITER_D0]], %[[J_SZ]]] [1, 1]
// CHECK:           %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:        dimension_map = [0] unique_indices(true)
// CHECK-SAME:        ins(%[[UPDATES_TILE]], %[[INDICES_TILE]]
// CHECK-SAME:       outs(%[[ITER_TILE]]
// CHECK:           %[[ORIGINAL_D0:.+]] = tensor.dim %[[ORIGINAL]], %[[C0]]
// CHECK:           scf.forall.in_parallel {
// CHECK:              tensor.parallel_insert_slice %[[SCATTER]] into %[[ITER]]
// CHECK-SAME:           [0, %[[J]]] [%[[ORIGINAL_D0]], %[[J_SZ]]]
// CHECK:           }
// CHECK          } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
// CHECK:         return %[[RESULT]]

// -----

func.func @sort_3d_multi_result_distribute(
  %arg0: tensor<?x?x?xi32>, %arg1 : tensor<?x?x?xf32>)
  -> (tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
  %0, %1 = iree_linalg_ext.sort
      dimension(2)
      outs(%arg0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %2 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %2 : i1
      } -> tensor<?x?x?xi32>, tensor<?x?x?xf32>
  return %0, %1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>
}
transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %0 = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %forall, %tiled_op = transform.structured.tile_using_forall %0 tile_sizes [10, 30, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 10)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 30)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0) -> (d0 * 30)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0)[s0] -> (d0 * -30 + s0, 30)>
// CHECK:       func.func @sort_3d_multi_result_distribute(
// CHECK-SAME:      %[[SRC0:[a-zA-Z0-9_]+]]
// CHECK-SAME:      %[[SRC1:[a-zA-Z0-9_]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[D0:.+]] = tensor.dim %[[SRC0]], %[[C0]]
// CHECK-DAG:     %[[D1:.+]] = tensor.dim %[[SRC0]], %[[C1]]
// CHECK-DAG:     %[[D2:.+]] = tensor.dim %[[SRC0]], %[[C2]]
// CHECK-DAG:     %[[UB0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
// CHECK-DAG:     %[[UB1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
// CHECK:         %[[RESULT:.+]]:2 = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[UB0]], %[[UB1]]) shared_outs(%[[ITER0:.+]] = %[[SRC0]], %[[ITER1:.+]] = %[[SRC1]])
// CHECK-DAG:       %[[I:.+]] = affine.apply #[[MAP2]](%[[IV0]])
// CHECK-DAG:       %[[I_SZ:.+]] = affine.min #[[MAP3]](%[[IV0]])[%[[D0]]]
// CHECK-DAG:       %[[J:.+]] = affine.apply #[[MAP4]](%[[IV1]])
// CHECK-DAG:       %[[J_SZ:.+]] = affine.min #[[MAP5]](%[[IV1]])[%[[D1]]]
// CHECK:           %[[ITER0_TILE:.+]] = tensor.extract_slice %[[ITER0]]
// CHECK-SAME:        [%[[I]], %[[J]], 0] [%[[I_SZ]], %[[J_SZ]], %[[D2]]]
// CHECK:           %[[ITER1_TILE:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:        [%[[I]], %[[J]], 0] [%[[I_SZ]], %[[J_SZ]], %[[D2]]]
// CHECK:           %[[SORT:.+]]:2 = iree_linalg_ext.sort dimension(2)
// CHECK-SAME:        outs(%[[ITER0_TILE]], %[[ITER1_TILE]]
// CHECK:           scf.forall.in_parallel {
// CHECK:             tensor.parallel_insert_slice %[[SORT]]#0 into %[[ITER0]]
// CHECK-SAME:          [%[[I]], %[[J]], 0] [%[[I_SZ]], %[[J_SZ]], %[[D2]]] [1, 1, 1]
// CHECK:             tensor.parallel_insert_slice %[[SORT]]#1 into %[[ITER1]]
// CHECK-SAME:          [%[[I]], %[[J]], 0] [%[[I_SZ]], %[[J_SZ]], %[[D2]]] [1, 1, 1]
// CHECK:           }
// CHECK          }
// CHECK:         return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @sort_3d_multi_result_distribute_memref(
  %arg0: memref<?x?x?xi32>, %arg1 : memref<?x?x?xf32>) {
  iree_linalg_ext.sort
      dimension(2)
      outs(%arg0, %arg1 : memref<?x?x?xi32>, memref<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %0 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %0 : i1
      }
  return
}
transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %0 = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %forall, %tiled_op = transform.structured.tile_using_forall %0 tile_sizes [10, 30, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 10)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 30)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0) -> (d0 * 30)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0)[s0] -> (d0 * -30 + s0, 30)>
// CHECK:       func.func @sort_3d_multi_result_distribute_memref(
// CHECK-SAME:      %[[SRC0:[a-zA-Z0-9_]+]]
// CHECK-SAME:      %[[SRC1:[a-zA-Z0-9_]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[SRC0]], %[[C0]]
// CHECK-DAG:     %[[D1:.+]] = memref.dim %[[SRC0]], %[[C1]]
// CHECK-DAG:     %[[D2:.+]] = memref.dim %[[SRC0]], %[[C2]]
// CHECK-DAG:     %[[UB0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
// CHECK-DAG:     %[[UB1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
// CHECK:         scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[UB0]], %[[UB1]])
// CHECK-DAG:       %[[I:.+]] = affine.apply #[[MAP2]](%[[IV0]])
// CHECK-DAG:       %[[I_SZ:.+]] = affine.min #[[MAP3]](%[[IV0]])[%[[D0]]]
// CHECK-DAG:       %[[J:.+]] = affine.apply #[[MAP4]](%[[IV1]])
// CHECK-DAG:       %[[J_SZ:.+]] = affine.min #[[MAP5]](%[[IV1]])[%[[D1]]]
// CHECK:           %[[SRC0_TILE:.+]] = memref.subview %[[SRC0]]
// CHECK-SAME:        [%[[I]], %[[J]], 0] [%[[I_SZ]], %[[J_SZ]], %[[D2]]]
// CHECK:           %[[SRC1_TILE:.+]] = memref.subview %[[SRC1]]
// CHECK-SAME:        [%[[I]], %[[J]], 0] [%[[I_SZ]], %[[J_SZ]], %[[D2]]]
// CHECK:           iree_linalg_ext.sort dimension(2)
// CHECK-SAME:        outs(%[[SRC0_TILE]], %[[SRC1_TILE]]
// CHECK          }
// CHECK:         return
