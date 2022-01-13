// RUN: iree-opt -split-input-file -verify-diagnostics -pass-pipeline="builtin.func(iree-flow-dispatch-linalg-on-tensors-pass, resolve-shaped-type-result-dims, cse, canonicalize, cse)" %s | FileCheck %s

func @tile_matmul_alone(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: #[[MULMAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: func @tile_matmul_alone
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[ARG0_DIM0:.+]] = tensor.dim %[[ARG0]], %c0
//  CHECK-DAG:   %[[ARG0_DIM1:.+]] = tensor.dim %[[ARG0]], %c1
//  CHECK-DAG:   %[[ARG1_DIM0:.+]] = tensor.dim %[[ARG1]], %c0
//  CHECK-DAG:   %[[ARG1_DIM1:.+]] = tensor.dim %[[ARG1]], %c1
//  CHECK-DAG:   %[[ARG2_DIM0:.+]] = tensor.dim %[[ARG2]], %c0
//  CHECK-DAG:   %[[ARG2_DIM1:.+]] = tensor.dim %[[ARG2]], %c1
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     (%[[ARG0]], %[[ARG0_DIM0]], %[[ARG0_DIM1]], %[[ARG1]], %[[ARG1_DIM0]], %[[ARG1_DIM1]], %[[ARG2]], %[[ARG2_DIM0]], %[[ARG2_DIM1]])
// CHECK-NEXT:     %[[ARG3:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
//  CHECK-DAG:     %[[WGSIZE_X:.+]] = flow.dispatch.workgroup.size[0]
//  CHECK-DAG:     %[[WGSIZE_Y:.+]] = flow.dispatch.workgroup.size[1]
//  CHECK-DAG:     %[[WGID_X:.+]] = flow.dispatch.workgroup.id[0]
//  CHECK-DAG:     %[[WGID_Y:.+]] = flow.dispatch.workgroup.id[1]
//  CHECK-DAG:     %[[WGCOUNT_X:.+]] = flow.dispatch.workgroup.count[0]
//  CHECK-DAG:     %[[WGCOUNT_Y:.+]] = flow.dispatch.workgroup.count[1]
//      CHECK:     %[[OFFSET_Y:.+]] = affine.apply #[[MULMAP]]()[%[[WGID_Y]], %[[WGSIZE_Y]]]
//      CHECK:     %[[STEP_Y:.+]] = affine.apply #[[MULMAP]]()[%[[WGCOUNT_Y]], %[[WGSIZE_Y]]]
//      CHECK:     scf.for %[[ARG7:.+]] = %[[OFFSET_Y]]
// CHECK-SAME:       to %{{.+}} step %[[STEP_Y]]
//      CHECK:       %[[OFFSET_X:.+]] = affine.apply #[[MULMAP]]()[%[[WGID_X]], %[[WGSIZE_X]]]
//      CHECK:       %[[STEP_X:.+]] = affine.apply #[[MULMAP]]()[%[[WGCOUNT_X]], %[[WGSIZE_X]]]
//      CHECK:       scf.for %[[ARG8:.+]] = %[[OFFSET_X]]
// CHECK-SAME:         to %{{.+}} step %[[STEP_X]]
//      CHECK:         %[[LHS:.+]] = flow.dispatch.tensor.load %[[ARG3]],
// CHECK-SAME:           offsets = [%[[ARG7]], 0]
//      CHECK:         %[[RHS:.+]] = flow.dispatch.tensor.load %[[ARG4]],
// CHECK-SAME:           offsets = [0, %[[ARG8]]]
//      CHECK:         %[[INIT:.+]] = flow.dispatch.tensor.load %[[ARG5]],
// CHECK-SAME:           offsets = [%[[ARG7]], %[[ARG8]]]
//      CHECK:         %[[RESULT:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:         flow.dispatch.tensor.store %[[RESULT]], %[[ARG5]]
// CHECK-SAME:           offsets = [%[[ARG7]], %[[ARG8]]]

// -----

func @tile_generic_op_alone(%A: tensor<?x?xf32>, %B: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %A, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins (%A, %B: tensor<?x?xf32>, tensor<?xf32>)
    outs (%0 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %2 = arith.addf %arg0, %arg1 : f32
        linalg.yield %2 : f32
    } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @tile_generic_op_alone
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     [%[[ARG0_D1]], %[[ARG0_D0]], %[[C1]]](%[[ARG0]], %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG1]], %[[ARG1_D0]])
// CHECK-NEXT:     %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>, %[[ARG0_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D1_CAPTURE:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:     %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?xf32>, %[[ARG1_D0_CAPTURE:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:     %[[RET0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>
//  CHECK-DAG:     %[[LOAD2:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?x?xf32>{%[[ARG0_D0_CAPTURE]], %[[ARG0_D1_CAPTURE]]}
//  CHECK-DAG:     %[[LOAD3:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?xf32>{%[[ARG1_D0_CAPTURE]]}
//  CHECK-DAG:     %[[INIT:.+]] = linalg.init_tensor
//      CHECK:     %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[LOAD2]], %[[LOAD3]] : tensor<?x?xf32>, tensor<?xf32>)
// CHECK-SAME:         outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:     flow.dispatch.tensor.store %[[RESULT]], %[[RET0_CAPTURE]], {{.*}} -> !flow.dispatch.tensor<writeonly:?x?xf32>{%[[ARG0_D0_CAPTURE]], %[[ARG0_D1_CAPTURE]]}

// -----

func @fuse_matmul_with_fill(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf32>
  %N = tensor.dim %B, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%M, %N] : tensor<?x?xf32>
  %1 = linalg.fill(%zero, %0) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %2 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//       CHECK:   func @fuse_matmul_with_fill
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[ARG0_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:     %[[ARG0_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:     %[[ARG1_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:     %[[ARG1_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-NEXT:     flow.dispatch.workgroups[%[[ARG1_DIM1]], %[[ARG0_DIM0]], %[[C1]]]
//  CHECK-SAME:       (%[[ARG0]], %[[ARG0_DIM0]], %[[ARG0_DIM1]], %[[ARG1]], %[[ARG1_DIM0]], %[[ARG1_DIM1]])
//  CHECK-NEXT:       (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>,
//  CHECK-SAME:        %[[ARG0_DIM0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_DIM1_CAPTURE:[a-zA-Z0-9_]+]]: index,
//  CHECK-SAME:        %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>,
//  CHECK-SAME:        %[[ARG1_DIM0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_DIM1_CAPTURE:[a-zA-Z0-9_]+]]: index,
//  CHECK-SAME:        %[[RET0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>) {
//       CHECK:        %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:        scf.for
//       CHECK:          scf.for
//   CHECK-DAG:            %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?x?xf32>{%[[ARG0_DIM0_CAPTURE]], %[[ARG0_DIM1_CAPTURE]]}
//   CHECK-DAG:            %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?x?xf32>{%[[ARG1_DIM0_CAPTURE]], %[[ARG1_DIM1_CAPTURE]]}
//   CHECK-DAG:            %[[INIT_TILE:.+]] = linalg.init_tensor
//       CHECK:            %[[FILL_TILE:.+]] = linalg.fill(%[[ZERO]], %[[INIT_TILE]])
//       CHECK:            %[[RESULT_TILE:.+]] = linalg.matmul
//  CHECK-SAME:              ins(%[[LHS_TILE]], %[[RHS_TILE]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:              outs(%[[FILL_TILE]] : tensor<?x?xf32>)
//       CHECK:            flow.dispatch.tensor.store %[[RESULT_TILE]], %[[RET0_CAPTURE]], {{.*}} -> !flow.dispatch.tensor<writeonly:?x?xf32>{%[[ARG0_DIM0_CAPTURE]], %[[ARG1_DIM1_CAPTURE]]}
//       CHECK:          flow.return
//       CHECK:        }

// -----

func @keep_separate_dispatches_for_producer(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf32>
  %N = tensor.dim %B, %c1 : tensor<?x?xf32>
  %K = tensor.dim %A, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%M, %N] : tensor<?x?xf32>
  %1 = linalg.fill(%zero, %0) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %2 = linalg.init_tensor [%M, %K] : tensor<?x?xf32>
  %3 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%A : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
      %4 = arith.addf %arg0, %one : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%3, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
//      CHECK: func @keep_separate_dispatches_for_producer
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:     %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:     %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:     %[[RESULT1:.+]] = flow.dispatch.workgroups[%[[K]], %[[M]], %[[C1]]]
//  CHECK-SAME:       (%[[ARG0]], %[[M]], %[[K]])
//  CHECK-NEXT:       (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  CHECK-SAME:        %[[RET0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>) {
//       CHECK:          %[[ONE:.+]] = arith.constant 1.0
//   CHECK-DAG:          %[[INPUT:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
//   CHECK-DAG:          %[[INIT:.+]] = linalg.init_tensor
//       CHECK:          %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:            ins(%[[INPUT]] : tensor<?x?xf32>)
//  CHECK-SAME:            outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:          flow.dispatch.tensor.store %[[RESULT]], %[[RET0_CAPTURE]]
//       CHECK:          flow.return
//       CHECK:     }
//       CHECK:     flow.dispatch.workgroups[%[[N]], %[[M]], %[[C1]]]
//       CHECK:       %[[ZERO:.+]] = arith.constant 0.0
//       CHECK:       scf.for
//       CHECK:         scf.for
//       CHECK:            %[[INIT_TILE:.+]] = linalg.init_tensor
//       CHECK:            %[[FILL_TILE:.+]] = linalg.fill(%[[ZERO]], %[[INIT_TILE]])
//       CHECK:            linalg.matmul
//       CHECK:              outs(%[[FILL_TILE]] : tensor<?x?xf32>)

// The following CHECK* sems to hit a segfault with FileCheck. For now using a simpler check.
//  NOCHECK-SAME:       (%[[M]], %[[N]], %[[ARG0]], %[[ARG1]], %[[RESULT1]])
//  NOCHECK-SAME:       (%[[ARG2:[a-zA-Z0-9_]+]]: index
//  NOCHECK-SAME:        %[[ARG3:[a-zA-Z0-9_]+]]: index
//  NOCHECK-SAME:        %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  NOCHECK-SAME:        %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  NOCHECK-SAME:        %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  NOCHECK-SAME:        %[[ARG7:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>) {
//       NOCHECK:          %[[ZERO:.+]] = arith.constant 0.0
//       NOCHECK:          scf.for
//       NOCHECK:            scf.for
//   NOCHECK-DAG:              %[[LHS_TILE_2:.+]] = flow.dispatch.tensor.load %[[ARG6]], {{.*}}
//   NOCHECK-DAG:              %[[RHS_TILE_2:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//   NOCHECK-DAG:              %[[INIT_TILE_2:.+]] = linalg.init_tensor
//       NOCHECK:              %[[FILL_TILE:.+]] = linalg.fill(%[[ZERO]], %[[INIT_TILE]])
//       NOCHECK:              %[[RESULT_TILE_2:.++]]] = linalg.matmul
//  NOCHECK-SAME:                ins(%[[LHS_TILE_2]], %[[RHS_TILE_2]] : tensor<?x?xf32>, tensor<?x?xf32>)
//       NOCHECK:                outs(%[[FILL_TILE_2]] : tensor<?x?xf32>)
//       NOCHECK:              flow.dispatch.tensor.store %[[RESULT_TILE_2]], %[[ARG7]]
//       NOCHECK:          flow.return
//       NOCHECK:        }

// -----

func @tile_4d_generic_op_alone
  (%A: tensor<?x?x?x?xf32>, %B: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %A, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %A, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %A, %c3 : tensor<?x?x?x?xf32>
  %0 = linalg.init_tensor [%d0, %d1, %d2, %d3] : tensor<?x?x?x?xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins (%A, %B: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%0 : tensor<?x?x?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %2 = arith.addf %arg0, %arg1 : f32
        linalg.yield %2 : f32
    } -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
// For ops of rank greater than 3 we serialized the higher dimension. When flow
// supports larger ranks this can be changed.
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: func @tile_4d_generic_op_alone
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[D3:.+]] = tensor.dim %[[ARG0]], %[[C3]]
//  CHECK-DAG:   %[[WG_SISE_2:.+]] = flow.dispatch.workgroup.size[2] : index
//  CHECK-DAG:   %[[WG_ID_2:.+]] = flow.dispatch.workgroup.id[2] : index
//  CHECK-DAG:   flow.dispatch.workgroups[%[[D2]], %[[D1]], %[[D0]]]
//  CHECK-DAG:   %[[D4:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_2]], %[[WG_SISE_2]]]

// -----

func @always_fuse_reshape
  (%lhs : tensor<?xf32>, %rhs1 : tensor<4x?xf32>, %rhs2 : tensor<4x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.expand_shape %lhs [[0, 1]]
    : tensor<?xf32> into tensor<?x4xf32>
  %m = tensor.dim %0, %c0 : tensor<?x4xf32>
  %n1 = tensor.dim %rhs1, %c1 : tensor<4x?xf32>
  %init1 = linalg.init_tensor [%m, %n1] : tensor<?x?xf32>
  %fill1 = linalg.fill(%cst, %init1) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = linalg.matmul
    ins(%0, %rhs1 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %n2 = tensor.dim %rhs2, %c1 : tensor<4x?xf32>
  %init2 = linalg.init_tensor [%m, %n2] : tensor<?x?xf32>
  %fill2 = linalg.fill(%cst, %init2) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %2= linalg.matmul
    ins(%0, %rhs2 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
//      CHECK: func @always_fuse_reshape(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK-SAME:   %[[RHS1:[a-zA-Z0-9_]+]]: tensor<4x?xf32>
// CHECK-SAME:   %[[RHS2:[a-zA-Z0-9_]+]]: tensor<4x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[M:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//  CHECK-DAG:   %[[N1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RESULT1:.+]] = flow.dispatch.workgroups[%[[N1]], %[[M]], %[[C1]]]
// CHECK-SAME:     (%[[RHS1]], %[[N1]], %[[ARG0]], %[[D0]], %[[M]])
//      CHECK:   %[[N2:.+]] = tensor.dim %[[RHS2]], %[[C1]]
//      CHECK:   %[[RESULT2:.+]] = flow.dispatch.workgroups[%[[N2]], %[[M]], %[[C1]]]
// CHECK-SAME:     (%[[RHS2]], %[[N2]], %[[ARG0]], %[[D0]], %[[M]])
//      CHECK:   return %[[RESULT1]], %[[RESULT2]]

// -----

// A subsequent pass is expected to convert linalg.fill and flow.tensor.update into DMA ops.
func @dont_fuse_tensor_update_with_fill(
    %arg0: tensor<?x?xf32>, %arg1: tensor<f32>,
    %arg2: index, %arg3: index, %arg4: index, %arg5: index)
-> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%1)[%arg2, %arg4]
  %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%2)[%arg3, %arg5]
  %5 = linalg.init_tensor [%3, %4] : tensor<?x?xf32>
  %6 = linalg.fill(%0, %5) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %7 = flow.tensor.update %arg0, %6[%arg2, %arg3] : tensor<?x?xf32>{%1, %2} -> %6 as tensor<?x?xf32>{%3, %4}
  return %7 : tensor<?x?xf32>
}

// CHECK: func @dont_fuse_tensor_update_with_fill
// CHECK:   linalg.fill
// CHECK:   flow.tensor.update

// -----

// CHECK-LABEL: func @pass_constant_through()
func @pass_constant_through() -> tensor<2x2x3xi32> {
  // CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<2x2x3xi32>
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  // CHECK: return %[[CST]]
  return %cst : tensor<2x2x3xi32>
}

// -----

// CHECK-LABEL: func @fuse_matmul_with_generic_op
func @fuse_matmul_with_generic_op(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> {
  %f12 = arith.constant 12.0 : f32

  // linalg.generic is fused inside the dispatch region and becomes dead.
  // CHECK-NOT: generic
  %CC = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%C : tensor<?x?xf32>) {
    ^bb0(%c: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  //     CHECK: flow.dispatch.workgroups
  // CHECK-NOT:   generic
  //     CHECK:   scf.for
  //     CHECK:     scf.for
  //     CHECK:       %[[CC:.*]] = linalg.generic
  //     CHECK:       linalg.matmul{{.*}} outs(%[[CC]]
  %D = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%CC: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}

// -----

//       CHECK: func @keep_original_producer_uses
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
func @keep_original_producer_uses(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %f12 = arith.constant 12.0 : f32
  //  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  //  CHECK-DAG: %[[D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
  //  CHECK-DAG: %[[D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
  //      CHECK: %[[origCC:.+]] = flow.dispatch.workgroups[%[[D1]], %[[D0]], %[[C1]]](%[[ARG2]], %[[D0]], %[[D1]])
  // CHECK-NEXT:   %[[ARG2_CAPTURE:.+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
  //      CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[ARG2_CAPTURE]], {{.*}}
  //      CHECK:   %[[STOREVAL:.+]] = linalg.generic
  // CHECK-SAME:     outs(%[[LOAD]] : tensor<?x?xf32>)
  //      CHECK:   flow.dispatch.tensor.store %[[STOREVAL]], %[[ARG2_CAPTURE]], {{.*}}

  // linalg.generic is fused inside the dispatch region and becomes a noop but
  // there is still a use.
  %CC = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%C : tensor<?x?xf32>) {
    ^bb0(%c: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  //     CHECK: %[[D:.*]] = flow.dispatch.workgroups
  // Check origCC is not an operand of flow.dispatch.workgroups
  // CHECK-NOT: %[[origCC]],
  // CHECK-NOT:   linalg.generic
  //     CHECK:   scf.for
  //     CHECK:     scf.for
  //     CHECK:       %[[CC:.*]] = linalg.generic
  //     CHECK:       linalg.matmul{{.*}} outs(%[[CC]]
  %D = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%CC: tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: return %[[D]], %[[origCC]]
  return %D, %CC: tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

func @conv2d(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill(%cst, %0) : f32, tensor<1x112x112x32xf32> -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_nhwc_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  return %2 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func @conv2d
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.conv_2d_nhwc_hwcf

// -----

func @depthwise_conv2d(%input: tensor<1x113x113x96xf32>, %filter: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
  %2 = linalg.fill(%cst, %1) : f32, tensor<1x56x56x96xf32> -> tensor<1x56x56x96xf32>
  %4 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%input, %filter : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%2 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  return %4 : tensor<1x56x56x96xf32>
}

// CHECK-LABEL: func @depthwise_conv2d
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc

// -----

func @subtensor_insert(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index) -> tensor<?x?xf32> {
  %0 = tensor.insert_slice %arg0 into
      %arg1[%arg2, %arg3] [%arg4, %arg5] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//      CHECK: func @subtensor_insert
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[D3:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[%[[D1]], %[[D0]], %[[C1]]]
// CHECK-SAME:       (%[[ARG1]], %[[ARG0]], %[[D0]], %[[D1]], %[[ARG2]], %[[ARG3]], %[[D2]], %[[D3]])
// CHECK-SAME:       tensor<?x?xf32>{%[[D2]], %[[D3]]}
// CHECK-SAME:       tensor<?x?xf32>{%[[D0]], %[[D1]]}
// CHECK-SAME:       -> %[[ARG1]]{%[[D2]], %[[D3]]}
// CHECK-NEXT:     %[[ARG1_CAPTURE:.+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
// CHECK-SAME:     %[[ARG0_CAPTURE:.+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[D0_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[D1_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[D2_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[D3_CAPTURE:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[WGSIZE_X:.+]] = flow.dispatch.workgroup.size[0]
//  CHECK-DAG:     %[[WGSIZE_Y:.+]] = flow.dispatch.workgroup.size[1]
//  CHECK-DAG:     %[[WGID_X:.+]] = flow.dispatch.workgroup.id[0]
//  CHECK-DAG:     %[[WGCOUNT_X:.+]] = flow.dispatch.workgroup.count[0]
//  CHECK-DAG:     %[[WGID_Y:.+]] = flow.dispatch.workgroup.id[1]
//  CHECK-DAG:     %[[WGCOUNT_Y:.+]] = flow.dispatch.workgroup.count[1]
//  CHECK-DAG:     %[[OFFSET_Y:.+]] = affine.apply #[[MAP0]](%[[WGID_Y]])[%[[WGSIZE_Y]]]
//  CHECK-DAG:     %[[STEP_Y:.+]] = affine.apply #[[MAP0]](%[[WGCOUNT_Y]])[%[[WGSIZE_Y]]]
//      CHECK:     scf.for %[[ARG10:.+]] = %[[OFFSET_Y]] to %[[D0_CAPTURE]] step %[[STEP_Y]]
//  CHECK-DAG:       %[[TILESIZE_Y:.+]] = affine.min #[[MAP1]](%[[ARG10]])[%[[WGSIZE_Y]], %[[D0_CAPTURE]]]
//  CHECK-DAG:       %[[OFFSET_X:.+]] = affine.apply #[[MAP0]](%[[WGID_X]])[%[[WGSIZE_X]]]
//  CHECK-DAG:       %[[STEP_X:.+]] = affine.apply #[[MAP0]](%[[WGCOUNT_X]])[%[[WGSIZE_X]]]
//      CHECK:       scf.for %[[ARG11:.+]] = %[[OFFSET_X]] to %[[D1_CAPTURE]] step %[[STEP_X]]
//      CHECK:         %[[TILESIZE_X:.+]] = affine.min #[[MAP1]](%[[ARG11]])[%[[WGSIZE_X]], %[[D1_CAPTURE]]]
//      CHECK:         %[[LOAD_TILE:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
// CHECK-SAME:             offsets = [%[[ARG10]], %[[ARG11]]], sizes = [%[[TILESIZE_Y]], %[[TILESIZE_X]]]
//  CHECK-DAG:         %[[STORE_OFFSET_Y:.+]] = affine.apply #[[MAP2]](%[[ARG10]])[%[[ARG2_CAPTURE]]]
//  CHECK-DAG:         %[[STORE_OFFSET_Y:.+]] = affine.apply #[[MAP2]](%[[ARG11]])[%[[ARG3_CAPTURE]]]
//      CHECK:         flow.dispatch.tensor.store %[[LOAD_TILE]], %[[ARG1_CAPTURE]]
//      CHECK:   return %[[RESULT]]

// -----

func @fuse_non_tiled_reduction_fill(%input1: tensor<1000xf32>, %input2: tensor<1000xf32>, %offset: tensor<f32>) -> tensor<f32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.init_tensor [] : tensor<f32>
  %fill = linalg.fill(%zero, %init) : f32, tensor<f32> -> tensor<f32>
  %reduce = linalg.generic {
              indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
              iterator_types = ["reduction"]}
            ins(%input1, %input2, %offset : tensor<1000xf32>, tensor<1000xf32>, tensor<f32>)
            outs(%fill : tensor<f32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    %555 = arith.addf %arg1, %arg2 : f32
    %556 = arith.subf %555, %arg3 : f32
    %557 = math.exp %556 : f32
    %558 = arith.addf %557, %arg4 : f32
    linalg.yield %558 : f32
  } -> tensor<f32>
  return %reduce : tensor<f32>
}

// CHECK-LABEL: func @fuse_non_tiled_reduction_fill

//      CHECK: %[[C1:.+]] = arith.constant 1 : index
//      CHECK: flow.dispatch.workgroups[%[[C1]], %[[C1]], %[[C1]]]({{.+}}) : (tensor<1000xf32>, tensor<1000xf32>, tensor<f32>) -> tensor<f32> =
// CHECK-NEXT:     (%[[INPUT1:[a-z0-9]+]]: !flow.dispatch.tensor<readonly:1000xf32>,
// CHECK-SAME:      %[[INPUT2:[a-z0-9]+]]: !flow.dispatch.tensor<readonly:1000xf32>,
// CHECK-SAME:      %[[OFFSET:[a-z0-9]+]]: !flow.dispatch.tensor<readonly:f32>,
// CHECK-SAME:      %[[OUTPUT:[a-z0-9]+]]: !flow.dispatch.tensor<writeonly:f32>) {
//  CHECK-DAG:   %[[INPUT1_LOAD:.+]] = flow.dispatch.tensor.load %[[INPUT1]], {{.*}}
//  CHECK-DAG:   %[[INPUT2_LOAD:.+]] = flow.dispatch.tensor.load %[[INPUT2]], {{.*}}
//  CHECK-DAG:   %[[OFFSET_LOAD:.+]] = flow.dispatch.tensor.load %[[OFFSET]], {{.*}}
//      CHECK:   %[[FILL:.+]] = linalg.fill
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:     ins(%[[INPUT1_LOAD]], %[[INPUT2_LOAD]], %[[OFFSET_LOAD]] : tensor<1000xf32>, tensor<1000xf32>, tensor<f32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<f32>)
//      CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUTPUT]], {{.*}}

// -----

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
func @inline_dag_1(
    %arg0: tensor<?xf32>, %arg1 : tensor<1x?xf32>, %arg2 : tensor<i32>,
    %arg3 : index) -> tensor<?xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
  %1 = tensor.extract_slice %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %3 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %4 = tensor.extract_slice %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = tensor.collapse_shape %4 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %6 = tensor.extract_slice %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = tensor.collapse_shape %6 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %8 = linalg.init_tensor [%arg3] : tensor<?xf32>
  %9 = linalg.generic {
      indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map1],
      iterator_types = ["parallel"]}
      ins(%arg2, %2, %3, %5, %7 : tensor<i32>, tensor<?xf32>,
          tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      outs(%8 : tensor<?xf32>) {
      ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32):
        %10 = arith.addf %arg5, %arg6 : f32
        %11 = arith.addf %arg7, %arg8 : f32
        %12 = arith.addf %10, %11 : f32
        %13 = arith.sitofp %arg4 : i32 to f32
        %14 = arith.addf %12, %13 : f32
        linalg.yield %14 : f32
      } -> tensor<?xf32>
  return %9 : tensor<?xf32>
}
// CHECK-LABEL: func @inline_dag_1
//   CHECK-NOT:   linalg.
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x?xf32>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?xf32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xf32>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG6]], {{.*}}
//       CHECK:     %[[OP1:.+]] = tensor.expand_shape %[[LEAF2]]
//       CHECK:     %[[OP2:.+]] = tensor.collapse_shape %[[LEAF1]]
//       CHECK:     %[[OP3:.+]] = tensor.extract_slice %[[OP1]][0, 0]
//       CHECK:     %[[OP4:.+]] = tensor.extract_slice %[[OP1]][0, 10]
//       CHECK:     %[[OP5:.+]] = tensor.extract_slice %[[OP1]][0, 20]
//       CHECK:     %[[OP6:.+]] = tensor.collapse_shape %[[OP3]]
//       CHECK:     %[[OP7:.+]] = tensor.collapse_shape %[[OP4]]
//       CHECK:     %[[OP8:.+]] = tensor.collapse_shape %[[OP5]]

// -----

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
func @inline_dag_2(
    %arg0: tensor<?xf32>, %arg1 : tensor<1x?xf32>, %arg2 : tensor<i32>,
    %arg3 : index) -> tensor<?xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
  %1 = tensor.extract_slice %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  br ^bb1
^bb1:
  %3 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %4 = tensor.extract_slice %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = tensor.collapse_shape %4 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %6 = tensor.extract_slice %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = tensor.collapse_shape %6 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %8 = linalg.init_tensor [%arg3] : tensor<?xf32>
  %9 = linalg.generic {
      indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map1],
      iterator_types = ["parallel"]}
      ins(%arg2, %3, %2, %5, %7 : tensor<i32>, tensor<?xf32>,
          tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      outs(%8 : tensor<?xf32>) {
      ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32):
        %10 = arith.addf %arg5, %arg6 : f32
        %11 = arith.addf %arg7, %arg8 : f32
        %12 = arith.addf %10, %11 : f32
        %13 = arith.sitofp %arg4 : i32 to f32
        %14 = arith.addf %12, %13 : f32
        linalg.yield %14 : f32
      } -> tensor<?xf32>
  return %9 : tensor<?xf32>
}
// CHECK-LABEL: func @inline_dag_2
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x?xf32>
//       CHECK:   flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x?xf32>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?xf32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xf32>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG6]], {{.*}}
//       CHECK:     %[[OP1:.+]] = tensor.expand_shape %[[LEAF2]]
//       CHECK:     %[[OP2:.+]] = tensor.collapse_shape %[[LEAF1]]
//       CHECK:     %[[OP3:.+]] = tensor.extract_slice %[[OP1]][0, 0]
//       CHECK:     %[[OP4:.+]] = tensor.extract_slice %[[OP1]][0, 10]
//       CHECK:     %[[OP5:.+]] = tensor.extract_slice %[[OP1]][0, 20]
//       CHECK:     %[[OP6:.+]] = tensor.collapse_shape %[[OP3]]
//       CHECK:     %[[OP7:.+]] = tensor.collapse_shape %[[OP4]]
//       CHECK:     %[[OP8:.+]] = tensor.collapse_shape %[[OP5]]

// -----

func @inline_dag_3(%240 : tensor<9xi32>, %244 : tensor<18xi32>, %247 : tensor<i32>) -> tensor<9xi1> {
  %c9 = arith.constant 9 : index
  %c5_i32 = arith.constant 5 : i32
  %c0_i32 = arith.constant 0 : i32
  %c9_i32 = arith.constant 9 : i32
  %245 = flow.tensor.update %240, %244[%c9] : tensor<9xi32> -> %244 as tensor<18xi32>
  %248 = tensor.extract %247[] : tensor<i32>
  %249 = arith.cmpi slt, %248, %c9_i32 : i32
  %250 = select %249, %248, %c9_i32 : i32
  %251 = arith.cmpi sgt, %250, %c0_i32 : i32
  %252 = select %251, %250, %c0_i32 : i32
  %253 = arith.index_cast %252 : i32 to index
  %254 = tensor.extract_slice %245[%253] [9] [1] : tensor<18xi32> to tensor<9xi32>
  %255 = linalg.init_tensor [9] : tensor<9xi1>
  %256 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%254 : tensor<9xi32>) outs(%255 : tensor<9xi1>) {
        ^bb0(%arg20: i32, %arg21: i1):  // no predecessors
          %849 = arith.cmpi eq, %arg20, %c5_i32 : i32
          linalg.yield %849 : i1
      } -> tensor<9xi1>
  return %256 : tensor<9xi1>
}

//       CHECK: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>

//       CHECK: func @inline_dag_3
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<9xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<18xi32>
//  CHECK-SAME:   %[[ARG2:.+]]: tensor<i32>
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.update %[[ARG0]], %[[ARG1]]
//       CHECK:   flow.dispatch.workgroups
//  CHECK-SAME:     (%[[ARG2]], %[[UPDATE]])
//  CHECK-NEXT:     (%[[ARG3:.+]]: !flow.dispatch.tensor<readonly:i32>,
//  CHECK-SAME:      %[[ARG4:.+]]: !flow.dispatch.tensor<readonly:18xi32>,
//   CHECK-DAG:     %[[C5:.+]] = arith.constant 5 : i32
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[C9:.+]] = arith.constant 9 : i32
//   CHECK-DAG:     %[[ARG3V:.+]] = flow.dispatch.tensor.load %[[ARG3]]
//   CHECK-DAG:     %[[EXTRACT:.+]] = tensor.extract %[[ARG3V]]
//   CHECK-DAG:     %[[CMP1:.+]] = arith.cmpi slt, %[[EXTRACT]]
//   CHECK-DAG:     %[[SELECT1:.+]] = select %[[CMP1]], %[[EXTRACT]], %[[C9]]
//   CHECK-DAG:     %[[CMP2:.+]] = arith.cmpi sgt, %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[SELECT2:.+]] = select %[[CMP2]], %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[INDEX_CAST:.+]] = arith.index_cast %[[SELECT2]]
//       CHECK:     scf.for %[[IV0:.+]] =
//       CHECK:       %[[OFFSET:.+]] = affine.apply #[[MAP0]](%[[IV0]])[%[[INDEX_CAST]]
//       CHECK:       %[[ARG4V:.+]] = flow.dispatch.tensor.load %[[ARG4]], offsets = [%[[OFFSET]]
//       CHECK:     flow.return

// -----

#map = affine_map<() -> ()>
func @inline_dag_4(%arg0: tensor<4xi32>, %arg1: tensor<i32>) -> tensor<i16> {
  %c3_i32 = arith.constant 3 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.extract %arg1[] : tensor<i32>
  %1 = arith.cmpi slt, %0, %c3_i32 : i32
  %2 = select %1, %0, %c3_i32 : i32
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
  %4 = select %3, %2, %c0_i32 : i32
  %5 = arith.index_cast %4 : i32 to index
  %6 = tensor.extract_slice %arg0[%5] [1] [1] : tensor<4xi32> to tensor<i32>
  br ^bb1
^bb1:  // pred: ^bb0
  %7 = linalg.init_tensor [] : tensor<i16>
  %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%6 : tensor<i32>) outs(%7 : tensor<i16>) {
  ^bb0(%arg2: i32, %arg3: i16):  // no predecessors
    %9 = arith.trunci %arg2 : i32 to i16
    linalg.yield %9 : i16
  } -> tensor<i16>
  return %8 : tensor<i16>
}
// CHECK-LABEL: func @inline_dag_4
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<4xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<i32>
//       CHECK:   flow.dispatch.workgroups
//  CHECK-SAME:     (%[[ARG0]], %[[ARG1]])
//  CHECK-NEXT:     (%[[ARG2:.+]]: !flow.dispatch.tensor<readonly:4xi32>
//  CHECK-SAME:      %[[ARG3:.+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:      %[[ARG4:.+]]: !flow.dispatch.tensor<writeonly:i16>
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : i32
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG2]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG3]], {{.*}}
//       CHECK:     %[[INIT:.+]] = linalg.init_tensor [] : tensor<i16>
//       CHECK:     %[[OP1:.+]] = tensor.extract %[[LEAF2]][] : tensor<i32>
//       CHECK:     %[[OP2:.+]] = arith.cmpi slt, %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP3:.+]] = select %[[OP2]], %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP4:.+]] = arith.cmpi sgt, %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP5:.+]] = select %[[OP4]], %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP6:.+]] = arith.index_cast %[[OP5]] : i32 to index
//       CHECK:     %[[OP7:.+]] = tensor.extract_slice %[[LEAF1]][%[[OP6]]] [1] [1] : tensor<4xi32> to tensor<i32>
//       CHECK:     %[[RES:.+]] = linalg.generi
//  CHECK-SAME:       ins(%[[OP7]] : tensor<i32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<i16>) {
//       CHECK:     ^bb0(%[[ARG5:.+]]: i32, %{{.+}}: i16):
//       CHECK:       %[[TRUNC:.+]] = arith.trunci %[[ARG5]] : i32 to i16
//       CHECK:       linalg.yield %[[TRUNC]] : i16
//       CHECK:     } -> tensor<i16>
//       CHECK:     flow.dispatch.tensor.store %[[RES]], %[[ARG4]]

// -----

func @multi_result(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
  %cmin = arith.constant -2147483648 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = linalg.init_tensor [%0] : tensor<?xi32>
  %2 = linalg.fill(%cmin, %1) : i32, tensor<?xi32> -> tensor<?xi32>
  %3 = linalg.fill(%c0_i32, %1) : i32, tensor<?xi32> -> tensor<?xi32>
  %4:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
      outs(%1, %2 : tensor<?xi32>, tensor<?xi32>) {
  ^bb0(%arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
    %5 = arith.cmpi sge, %arg2, %arg4 : i32
    %6 = select %5, %arg2, %arg4 : i32
    %7 = arith.cmpi eq, %arg2, %arg4 : i32
    %8 = arith.cmpi slt, %arg3, %arg5 : i32
    %9 = select %8, %arg3, %arg5 : i32
    %10 = select %5, %arg3, %arg5 : i32
    %11 = select %7, %9, %10 : i32
    linalg.yield %6, %11 : i32, i32
  } -> (tensor<?xi32>, tensor<?xi32>)
  return %4#0, %4#1 : tensor<?xi32>, tensor<?xi32>
}
// CHECK-LABEL: func @multi_result
//       CHECK:   %[[RESULT:.+]]:2 = flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xi32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xi32>
//       CHECK:     scf.for
//       CHECK:       %[[TILED_RESULT:.+]]:2 = linalg.generic
//   CHECK-DAG:       flow.dispatch.tensor.store %[[TILED_RESULT]]#0, %[[ARG5]]
//   CHECK-DAG:       flow.dispatch.tensor.store %[[TILED_RESULT]]#1, %[[ARG6]]
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func @dynamic_slice(%arg0: tensor<?x?xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3 : index) -> tensor<1x?xi32> {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.extract %arg1[] : tensor<i32>
  %1 = arith.cmpi slt, %0, %c1_i32 : i32
  %2 = select %1, %0, %c1_i32 : i32
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
  %4 = select %3, %2, %c0_i32 : i32
  %5 = arith.index_cast %4 : i32 to index
  %6 = tensor.extract %arg2[] : tensor<i32>
  %7 = arith.cmpi slt, %6, %c0_i32 : i32
  %8 = select %7, %6, %c0_i32 : i32
  %9 = arith.cmpi sgt, %8, %c0_i32 : i32
  %10 = select %9, %8, %c0_i32 : i32
  %11 = arith.index_cast %10 : i32 to index
  %12 = tensor.extract_slice %arg0[%5, %11] [1, %arg3] [1, 1] : tensor<?x?xi32> to tensor<1x?xi32>
  return %12 : tensor<1x?xi32>
}
// CHECK-LABEL: func @dynamic_slice(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<i32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<i32>
//  CHECK-SAME:   %[[ARG3:.+]]: index
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %c0
//   CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %c1
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:     [%[[ARG3]], %[[C1]], %[[C1]]]
//  CHECK-SAME:     (%[[ARG3]], %[[ARG1]], %[[ARG2]], %[[ARG0]], %[[ARG0_D0]], %[[ARG0_D1]])
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     select
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     select
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     select
//   CHECK-DAG:     select
//   CHECK-DAG:     index_cast
//   CHECK-DAG:     index_cast
//   CHECK-NOT:     tensor.extract
//       CHECK:     scf.for
//       CHECK:       scf.for
//   CHECK-NOT:         tensor.extract
//       CHECK:         flow.dispatch.tensor.load
//   CHECK-NOT:         tensor.extract
//       CHECK:         flow.dispatch.tensor.store
//       CHECK:     flow.return
//       CHECK:   return %[[RESULT]]

// -----

func @dynamic_dot() -> !hal.buffer_view attributes {iree.abi.stub} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = flow.tensor.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01], [1.200000e+01, 1.100000e+01, 1.000000e+01], [9.000000e+00, 8.000000e+00, 7.000000e+00], [6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<5x3xf32> -> tensor<?x?xf32>
  %1 = flow.tensor.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01, 1.200000e+01, 1.100000e+01], [1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<3x5xf32> -> tensor<?x?xf32>
  %2 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %3 = tensor.dim %1, %c1 : tensor<?x?xf32>
  %4 = linalg.init_tensor [%2, %3] : tensor<?x?xf32>
  %5 = linalg.fill(%cst, %4) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %6 = linalg.matmul ins(%0, %1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = tensor.dim %6, %c0 : tensor<?x?xf32>
  %8 = tensor.dim %6, %c1 : tensor<?x?xf32>
  %9 = hal.tensor.export %6 : tensor<?x?xf32>{%7, %8} -> !hal.buffer_view
  return %9 : !hal.buffer_view
}
// CHECK-LABEL: func @dynamic_dot()
//   CHECK-NOT:    linalg.fill
//   CHECK-NOT:    linalg.matmul
//       CHECK:    scf.for
//       CHECK:      scf.for
//       CHECK:        linalg.fill
//       CHECK:        linalg.matmul
//   CHECK-NOT:    linalg.fill
//   CHECK-NOT:    linalg.matmul
//       CHECK:    return

// -----

func @scatter(
    %original : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %1 = arith.addf %arg0, %arg1 : f32
        iree_linalg_ext.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//      CHECK: func @scatter(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[ARG2_D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[ARG2_D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//  CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[%[[ARG2_D1]], %[[ARG2_D0]], %[[C1]]]
// CHECK-SAME:       (%[[ARG2]], %[[ARG2_D0]], %[[ARG2_D1]], %[[ARG1]], %[[ARG1_D0]], %[[ARG0]], %[[ARG0_D0]], %[[ARG0_D1]])
// CHECK-NEXT:       %[[ARG2_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:       %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x1xi32>
// CHECK-SAME:       %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
//  CHECK-DAG:       %[[WGSIZE_X:.+]] = flow.dispatch.workgroup.size[0]
//  CHECK-DAG:       %[[WGSIZE_Y:.+]] = flow.dispatch.workgroup.size[1]
//  CHECK-DAG:       %[[WGID_X:.+]] = flow.dispatch.workgroup.id[0]
//  CHECK-DAG:       %[[WGCOUNT_X:.+]] = flow.dispatch.workgroup.count[0]
//  CHECK-DAG:       %[[WGID_Y:.+]] = flow.dispatch.workgroup.id[1]
//  CHECK-DAG:       %[[WGCOUNT_Y:.+]] = flow.dispatch.workgroup.count[1]
//  CHECK-DAG:       %[[LBY:.+]] = affine.apply #[[MAP0]](%[[WGID_Y]])[%[[WGSIZE_Y]]]
//  CHECK-DAG:       %[[STEPY:.+]] = affine.apply #[[MAP0]](%[[WGCOUNT_Y]])[%[[WGSIZE_Y]]]
//      CHECK:       scf.for %[[IV0:.+]] = %[[LBY]] to %{{.+}} step %[[STEPY]]
//  CHECK-DAG:         %[[LBX:.+]] = affine.apply #[[MAP0]](%[[WGID_X]])[%[[WGSIZE_X]]]
//  CHECK-DAG:         %[[STEPX:.+]] = affine.apply #[[MAP0]](%[[WGCOUNT_X]])[%[[WGSIZE_X]]]
//      CHECK:         scf.for %[[IV1:.+]] = %[[LBX]] to %{{.+}} step %[[STEPX]]
//  CHECK-DAG:           %[[UPDATE_TILE:.+]] = flow.dispatch.tensor.load %[[ARG2_CAPTURE]], offsets = [%[[IV0]], %[[IV1]]]
//  CHECK-DAG:           %[[INDICES_TILE:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], offsets = [%[[IV0]], 0]
//  CHECK-DAG:           %[[ORIGINAL_TILE:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], offsets = [0, %[[IV1]]]
//  CHECK-DAG:           %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:               ins(%[[UPDATE_TILE]], %[[INDICES_TILE]] : tensor<?x?xf32>, tensor<?x1xi32>)
// CHECK-SAME:               outs(%[[ORIGINAL_TILE]] : tensor<?x?xf32>)
//      CHECK:           flow.dispatch.tensor.store %[[SCATTER_TILE]], %[[ARG0_CAPTURE]], offsets = [0, %[[IV1]]]
//      CHECK:   return %[[RESULT]] : tensor<?x?xf32>

// -----

func @sort_3d(%arg0: tensor<?x?x?xi32>, %arg1 : tensor<?x?x?xf32>)
    -> (tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
  %0, %1 = iree_linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %2 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %2 : i1
      } -> tensor<?x?x?xi32>, tensor<?x?x?xf32>
  return %0, %1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MULMAP:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
//  CHECK-DAG: #[[MINMAP:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//      CHECK: func @sort_3d(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[ARG0_D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[ARG1_D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[ARG1_D2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//      CHECK:   %[[RESULT:.+]]:2 = flow.dispatch.workgroups[%[[ARG0_D2]], %[[ARG0_D1]], %[[C1]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG0_D2]], %[[ARG1]], %[[ARG1_D0]], %[[ARG1_D1]], %[[ARG1_D2]])
// CHECK-NEXT:       (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?x?x?xi32>
// CHECK-SAME:        %[[ARG0_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D1_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D2_CAPTURE:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:        %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?x?x?xf32>,
// CHECK-SAME:        %[[ARG1_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_D1_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_D2_CAPTURE:[a-zA-Z0-9_]+]]: index) {
//  CHECK-DAG:     %[[WGSIZE_Y:.+]] = flow.dispatch.workgroup.size[1]
//  CHECK-DAG:     %[[WGSIZE_X:.+]] = flow.dispatch.workgroup.size[0]
//  CHECK-DAG:     %[[WGID_X:.+]] = flow.dispatch.workgroup.id[0]
//  CHECK-DAG:     %[[WGCOUNT_X:.+]] = flow.dispatch.workgroup.count[0]
//  CHECK-DAG:     %[[WGID_Y:.+]] = flow.dispatch.workgroup.id[1]
//  CHECK-DAG:     %[[WGCOUNT_Y:.+]] = flow.dispatch.workgroup.count[1]
//  CHECK-DAG:     %[[LB_Y:.+]] = affine.apply #[[MULMAP]](%[[WGID_Y]])[%[[WGSIZE_Y]]]
//  CHECK-DAG:     %[[STEP_Y:.+]] = affine.apply #[[MULMAP]](%[[WGCOUNT_Y]])[%[[WGSIZE_Y]]]
//      CHECK:     scf.for %[[IV0:.+]] = %[[LB_Y]] to %[[ARG0_D1_CAPTURE]] step %[[STEP_Y]] {
//  CHECK-DAG:       %[[TS_Y:.+]] = affine.min #[[MINMAP]](%[[IV0]])[%[[WGSIZE_Y]], %[[ARG0_D1_CAPTURE]]]
//  CHECK-DAG:       %[[LB_X:.+]] = affine.apply #[[MULMAP]](%[[WGID_X]])[%[[WGSIZE_X]]]
//  CHECK-DAG:       %[[STEP_X:.+]] = affine.apply #[[MULMAP]](%[[WGCOUNT_X]])[%[[WGSIZE_X]]]
//      CHECK:       scf.for %[[IV1:.+]] = %[[LB_X]] to %[[ARG0_D2_CAPTURE]] step %[[STEP_X]] {
//      CHECK:         %[[TS_X:.+]] = affine.min #[[MINMAP]](%[[IV1]])[%[[WGSIZE_X]], %[[ARG0_D2_CAPTURE]]]
//  CHECK-DAG:         %[[OUT1_TILE:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
// CHECK-SAME:             offsets = [0, %[[IV0]], %[[IV1]]]
// CHECK-SAME:             sizes = [%[[ARG0_D0_CAPTURE]], %[[TS_Y]], %[[TS_X]]]
//  CHECK-DAG:         %[[OUT2_TILE:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]]
// CHECK-SAME:             offsets = [0, %[[IV0]], %[[IV1]]]
// CHECK-SAME:             sizes = [%[[ARG0_D0_CAPTURE]], %[[TS_Y]], %[[TS_X]]]
//      CHECK:         %[[RESULT_TILE:.+]]:2 = iree_linalg_ext.sort dimension(0)
// CHECK-SAME:             outs(%[[OUT1_TILE]], %[[OUT2_TILE]] :   tensor<?x?x?xi32>, tensor<?x?x?xf32>)
//  CHECK-DAG:         flow.dispatch.tensor.store %[[RESULT_TILE]]#0
// CHECK-SAME:             offsets = [0, %[[IV0]], %[[IV1]]]
// CHECK-SAME:             sizes = [%[[ARG0_D0_CAPTURE]], %[[TS_Y]], %[[TS_X]]]
//  CHECK-DAG:         flow.dispatch.tensor.store %[[RESULT_TILE]]#1
// CHECK-SAME:             offsets = [0, %[[IV0]], %[[IV1]]]
// CHECK-SAME:             sizes = [%[[ARG0_D0_CAPTURE]], %[[TS_Y]], %[[TS_X]]]
//      CHECK:       }
//      CHECK:     }
//      CHECK:     flow.return
//      CHECK:   }
//      CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func @sort_1d(%arg0: tensor<?xi32>, %arg1 : tensor<?xf32>)
    -> (tensor<?xi32>, tensor<?xf32>) {
  %0, %1 = iree_linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?xi32>, tensor<?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %2 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %2 : i1
      } -> tensor<?xi32>, tensor<?xf32>
  return %0, %1 : tensor<?xi32>, tensor<?xf32>
}
//      CHECK: func @sort_1d(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %c0
//  CHECK-DAG:   %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %c0
//      CHECK:   %[[RESULT:.+]]:2 = flow.dispatch.workgroups[%[[C1]], %[[C1]], %[[C1]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG0_D0]], %[[ARG1]], %[[ARG1_D0]])
// CHECK-NEXT:       (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?xi32>
// CHECK-SAME:        %[[ARG0_D0_CAPTURE:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:        %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?xf32>
// CHECK-SAME:        %[[ARG1_D0_CAPTURE:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:     %[[OUT1_TILE:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], offsets = [0], sizes = [%[[ARG0_D0_CAPTURE]]]
//  CHECK-DAG:     %[[OUT2_TILE:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], offsets = [0], sizes = [%[[ARG1_D0_CAPTURE]]]
//      CHECK:     %[[RESULT_TILE:.+]]:2 = iree_linalg_ext.sort dimension(0)
// CHECK-SAME:         outs(%[[OUT1_TILE]], %[[OUT2_TILE]] : tensor<?xi32>, tensor<?xf32>)
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT_TILE]]#0, %[[ARG0_CAPTURE]]
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT_TILE]]#1, %[[ARG1_CAPTURE]]
//      CHECK:     flow.return
//      CHECK:   }
//      CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func @scatter_static(%arg0 : tensor<4xi32>, %arg1 : tensor<4x1xi32>, %arg2 : tensor<8xi32>)
    -> tensor<8xi32>{
  %cst = arith.constant dense<[0, 9, 0, 10, 11, 0, 0, 12]> : tensor<8xi32>
  %cst_0 = arith.constant dense<[9, 10, 11, 12]> : tensor<4xi32>
  %cst_1 = arith.constant dense<[[1], [3], [4], [7]]> : tensor<4x1xi32>
  %cst_2 = arith.constant dense<0> : tensor<8xi32>
  %0 = iree_linalg_ext.scatter
      ins(%arg0, %arg1 : tensor<4xi32>, tensor<4x1xi32>)
      outs(%arg2 : tensor<8xi32>)  {
    ^bb0(%arg3: i32, %arg4: i32):  // no predecessors
      iree_linalg_ext.yield %arg3 : i32
    } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
//      CHECK: func @scatter_static
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4xi32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<4x1xi32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<8xi32>
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
// CHECK-NEXT:     %[[ARG3:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:4xi32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:4x1xi32>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:8xi32>
//      CHECK:     scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
//      CHECK:       %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
//      CHECK:       flow.dispatch.tensor.store %[[SCATTER_TILE]], %[[ARG5]], offsets = [0], sizes = [8], strides = [1]
// CHECK-NEXT:     }
//      CHECK:  return %[[RESULT]]

// -----

// Check that we are distributing along the last three dimensions for NHWC-output pooling op.

func @pooling_nwhc_sum_static(%input: tensor<1x33x33x160xf32>) -> tensor<1x1x1x160xf32> {
  %cst = arith.constant 0.0 : f32
  %1 = linalg.init_tensor [1, 1, 1, 160] : tensor<1x1x1x160xf32>
  %2 = linalg.fill(%cst, %1) : f32, tensor<1x1x1x160xf32> -> tensor<1x1x1x160xf32>
  %3 = linalg.init_tensor [33, 33] : tensor<33x33xf32>
  %4 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<33> : vector<2xi64>} ins(%input, %3 : tensor<1x33x33x160xf32>, tensor<33x33xf32>) outs(%2 : tensor<1x1x1x160xf32>) -> tensor<1x1x1x160xf32>
  return %4 : tensor<1x1x1x160xf32>
}

// CHECK-LABEL: func @pooling_nwhc_sum_static
//       CHECK:   flow.dispatch.workgroups
//  CHECK-NEXT:   (%{{.+}}: !flow.dispatch.tensor<readonly:1x33x33x160xf32>, %[[OUTPUT:.+]]: !flow.dispatch.tensor<writeonly:1x1x1x160xf32>)
//       CHECK:     scf.for %[[X:.+]] =
//       CHECK:       %[[POOL:.+]] = linalg.pooling_nhwc_sum
//       CHECK:       flow.dispatch.tensor.store %[[POOL]], %[[OUTPUT]], offsets = [0, 0, 0, %[[X]]], sizes = [1, 1, 1, %{{.+}}]

// -----

func @named_op_outs_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst1 = arith.constant -1.0 : f64
  %cstm1 = arith.constant 1.0 : f64
  %c12345 = arith.constant 12345 : i32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill_rng_2d ins(%cst1, %cstm1, %c12345 : f64, f64, i32)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %matmul = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %matmul : tensor<?x?xf32>
}
// CHECK-LABEL: func @named_op_outs_fusion
//       CHECK:   flow.dispatch.workgroups
//       CHECK:     %[[FILL:.+]] = linalg.fill_rng_2d
//       CHECK:     linalg.matmul
//  CHECK-SAME:       outs(%[[FILL]] : tensor<?x?xf32>)

// -----

func @dynamic_slice(%arg0 : i32, %arg1 : i32, %arg2 : tensor<?xi32>,
    %arg3 : tensor<?x?xi32>) -> tensor<?x?xi32>{
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %5 = arith.cmpi slt, %arg0, %c2_i32 : i32
  %6 = select %5, %arg0, %c2_i32 : i32
  %7 = arith.cmpi sgt, %6, %c0_i32 : i32
  %8 = select %7, %6, %c0_i32 : i32
  %9 = arith.index_cast %8 : i32 to index
  %11 = arith.cmpi slt, %arg1, %c0_i32 : i32
  %12 = select %11, %arg1, %c0_i32 : i32
  %13 = arith.cmpi sgt, %12, %c0_i32 : i32
  %14 = select %13, %12, %c0_i32 : i32
  %15 = arith.index_cast %14 : i32 to index
  %d0 = tensor.dim %arg2, %c0 : tensor<?xi32>
  %17 = tensor.insert_slice %arg2 into
      %arg3[%9, %15] [1, %d0] [1, 1] : tensor<?xi32> into tensor<?x?xi32>
  return %17 : tensor<?x?xi32>
}
// CHECK-LABEL: func @dynamic_slice
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: i32
//  CHECK-SAME:     %[[ARG2:.+]]: tensor<?xi32>
//  CHECK-SAME:     %[[ARG3:.+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG3]], %[[C0]]
//   CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG3]], %[[C1]]
//       CHECK:   flow.dispatch.workgroups[%[[D0]], %[[C1]], %[[C1]]]
//  CHECK-SAME:       tensor<?x?xi32>{%[[D1]], %[[D2]]}, tensor<?xi32>{%[[D0]]}
//  CHECK-NEXT:     %[[ARG4:.+]]: !flow.dispatch.tensor<readwrite:?x?xi32>
//  CHECK-SAME:     %[[ARG5:.+]]: !flow.dispatch.tensor<readonly:?xi32>

// -----

func @extract_slice(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index,
    %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> tensor<?x?xf32> {
  %0 = tensor.extract_slice %arg0[%arg1, %arg2] [%arg3, %arg4] [%arg5, %arg6] :
      tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: flow.dispatch.workgroups
// CHECK-NEXT:   %[[INPUT:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>
//      CHECK:   %[[SLICE:.+]] = flow.dispatch.tensor.load %[[INPUT]]
//      CHECK:   flow.dispatch.tensor.store %[[SLICE]], %[[OUTPUT]]

// -----

// TODO(ravishankarm): Enable after upstream pad op tiling issues are addressed.
// func @pad_tensor(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index,
//     %arg3 : index, %arg4 : index, %arg5 : f32) -> tensor<?x?xf32> {
//   %0 = tensor.pad %arg0 low[%arg1, %arg2] high[%arg3, %arg4] {
//     ^bb0(%arg6 : index, %arg7 : index):
//       tensor.yield %arg5 : f32
//   } :  tensor<?x?xf32> to tensor<?x?xf32>
//   return %0 : tensor<?x?xf32>
// }

// -----

func @inline_cst(%arg0 : tensor<4x32xi32>) -> tensor<32xi32> {
  %cst = arith.constant dense<0> : tensor<32xi32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<4x32xi32>) outs(%cst : tensor<32xi32>) {
      ^bb0(%arg1 : i32, %arg2 : i32) :
        %1 = arith.addi %arg1, %arg2 : i32
        linalg.yield %1 : i32
      } -> tensor<32xi32>
  return %0 : tensor<32xi32>
}
//      CHECK: func @inline_cst(%[[ARG0:.+]]: tensor<4x32xi32>)
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     (%[[ARG0]])
//      CHECK:     %[[CST:.+]] = arith.constant dense<0> : tensor<32xi32>

// -----

func @inline_cst2(%arg0 : tensor<4x2xi32>) -> tensor<2xi32> {
  %cst = arith.constant dense<[21, 42]> : tensor<2xi32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<4x2xi32>) outs(%cst : tensor<2xi32>) {
      ^bb0(%arg1 : i32, %arg2 : i32) :
        %1 = arith.addi %arg1, %arg2 : i32
        linalg.yield %1 : i32
      } -> tensor<2xi32>
  return %0 : tensor<2xi32>
}
//      CHECK: func @inline_cst2(%[[ARG0:.+]]: tensor<4x2xi32>)
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     (%[[ARG0]])
//      CHECK:     %[[CST:.+]] = arith.constant dense<[21, 42]> : tensor<2xi32>

// -----

func @gemm_unitN(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x1xf32>,
    %arg2 : tensor<?x1xf32>) -> tensor<?x1xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x1xf32>)
      outs(%arg2 : tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}
//      CHECK: func @gemm_unitN(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x1xf32>,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x1xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0:.+]], %[[C0]]
//      CHECK:   flow.dispatch.workgroups[%[[M]], %[[C1]], %[[C1]]]
//      CHECK:     scf.for
//  CHECK-NOT:       scf.for
//      CHECK:       linalg.matmul

// -----

func @gemm_unitM_unitN(%arg0 : tensor<1x1xf32>, %arg1 : tensor<1x1xf32>,
    %arg2 : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%arg2 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
//     CHECK: func @gemm_unitM_unitN(
//     CHECK:   %[[C1:.+]] = arith.constant 1 : index
//     CHECK:   flow.dispatch.workgroups[%[[C1]], %[[C1]], %[[C1]]]
// CHECK-NOT:     scf.for
//     CHECK:     linalg.matmul

// -----

func @gemm_unitM(%arg0 : tensor<1x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<1x?xf32>) -> tensor<1x?xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<1x?xf32>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}
//     CHECK: func @gemm_unitM(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xf32>,
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1:.+]], %[[C1]]
//      CHECK:   flow.dispatch.workgroups[%[[N]], %[[C1]], %[[C1]]]
//      CHECK:     scf.for
//  CHECK-NOT:       scf.for
//      CHECK:       linalg.matmul

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func @unit_dim_generic(%arg0 : tensor<1x?x1x1x?x?x1x?xf32>,
    %arg1 : tensor<1x?x1x1x?x?x1x?xf32>) -> tensor<1x?x1x1x?x?x1x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<1x?x1x1x?x?x1x?xf32>, tensor<1x?x1x1x?x?x1x?xf32>)
      outs(%arg0 : tensor<1x?x1x1x?x?x1x?xf32>) {
      ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
        %1 = arith.addf %arg2, %arg3 : f32
        linalg.yield %1 : f32
      } -> tensor<1x?x1x1x?x?x1x?xf32>
  return %0 : tensor<1x?x1x1x?x?x1x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0, -d1 + s0)>
//      CHECK: func @unit_dim_generic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?x1x1x?x?x1x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x?x1x1x?x?x1x?xf32>)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D4:.+]] = tensor.dim %[[ARG0]], %[[C4]]
//  CHECK-DAG:   %[[D5:.+]] = tensor.dim %[[ARG0]], %[[C5]]
//      CHECK:   flow.dispatch.workgroups[%[[D5]], %[[D4]], %[[D1]]]
// CHECK-SAME:       (%[[ARG0]], %[[D1]], %[[D4]], %[[D5]]
// CHECK-NEXT:      %[[ARG2:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:1x?x1x1x?x?x1x?xf32>
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[WG_SIZE_X:.+]] = flow.dispatch.workgroup.size[0] : index
//  CHECK-DAG:     %[[WG_SIZE_Y:.+]] = flow.dispatch.workgroup.size[1] : index
//  CHECK-DAG:     %[[WG_SIZE_Z:.+]] = flow.dispatch.workgroup.size[2] : index
//  CHECK-DAG:     %[[WG_ID_X:.+]] = flow.dispatch.workgroup.id[0] : index
//  CHECK-DAG:     %[[WG_COUNT_X:.+]] = flow.dispatch.workgroup.count[0] : index
//  CHECK-DAG:     %[[WG_ID_Y:.+]] = flow.dispatch.workgroup.id[1] : index
//  CHECK-DAG:     %[[WG_COUNT_Y:.+]] = flow.dispatch.workgroup.count[1] : index
//  CHECK-DAG:     %[[WG_ID_Z:.+]] = flow.dispatch.workgroup.id[2] : index
//  CHECK-DAG:     %[[WG_COUNT_Z:.+]] = flow.dispatch.workgroup.count[2] : index
//  CHECK-DAG:     %[[LB_Z:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Z]], %[[WG_SIZE_Z]]]
//  CHECK-DAG:     %[[STEP_Z:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_Z]], %[[WG_SIZE_Z]]]
//      CHECK:     scf.for %[[IV0:.+]] = %[[LB_Z]] to %[[ARG3]] step %[[STEP_Z]]
//  CHECK-DAG:       %[[LB_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Y]], %[[WG_SIZE_Y]]]
//  CHECK-DAG:       %[[STEP_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_Y]], %[[WG_SIZE_Y]]]
//      CHECK:       scf.for %[[IV1:.+]] = %[[LB_Y]] to %[[ARG4]] step %[[STEP_Y]]
//  CHECK-DAG:         %[[LB_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_X]], %[[WG_SIZE_X]]]
//  CHECK-DAG:         %[[STEP_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_X]], %[[WG_SIZE_X]]]
//      CHECK:         scf.for %[[IV2:.+]] = %[[LB_X]] to %[[ARG5]] step %[[STEP_X]]
//  CHECK-DAG:           %[[TILE_Z:.+]] = affine.min #[[MAP1]](%[[WG_SIZE_Z]], %[[IV0]])[%[[ARG3]]]
//  CHECK-DAG:           %[[TILE_Y:.+]] = affine.min #[[MAP1]](%[[WG_SIZE_Y]], %[[IV1]])[%[[ARG4]]]
//  CHECK-DAG:           %[[TILE_X:.+]] = affine.min #[[MAP1]](%[[WG_SIZE_X]], %[[IV2]])[%[[ARG5]]]
//      CHECK:           flow.dispatch.tensor.load %[[ARG2]]
// CHECK-SAME:               offsets = [0, %[[IV0]], 0, 0, %[[IV1]], %[[IV2]], 0, 0]
// CHECK-SAME:               sizes = [1, %[[TILE_Z]], 1, 1, %[[TILE_Y]], %[[TILE_X]], 1, %{{.+}}]
