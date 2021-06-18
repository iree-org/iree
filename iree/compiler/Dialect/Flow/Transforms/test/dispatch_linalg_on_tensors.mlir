// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-dispatch-linalg-on-tensors-pass -resolve-shaped-type-result-dims -canonicalize -cse %s | IreeFileCheck %s

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
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     (%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-NEXT:     %[[ARG3:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>
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
//      CHECK:         flow.dispatch.tensor.store %[[RESULT]], %[[ARG6]]
// CHECK-SAME:           offsets = [%[[ARG7]], %[[ARG8]]]

// -----

func @tile_generic_op_alone(%A: tensor<?x?xf32>, %B: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = memref.dim %A, %c0 : tensor<?x?xf32>
  %d1 = memref.dim %A, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins (%A, %B: tensor<?x?xf32>, tensor<?xf32>)
    outs (%0 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %2 = addf %arg0, %arg1 : f32
        linalg.yield %2 : f32
    } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @tile_generic_op_alone
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     [%[[D1]], %[[D0]], %[[C1]]](%[[ARG0]], %[[ARG1]], %[[D0]], %[[D1]])
// CHECK-NEXT:     %[[ARG2:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>
//  CHECK-DAG:     %[[LOAD2:.+]] = flow.dispatch.tensor.load %[[ARG2]], {{.*}}
//  CHECK-DAG:     %[[LOAD3:.+]] = flow.dispatch.tensor.load %[[ARG3]], {{.*}}
//  CHECK-DAG:     %[[INIT:.+]] = linalg.init_tensor
//      CHECK:     %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[LOAD2]], %[[LOAD3]] : tensor<?x?xf32>, tensor<?xf32>)
// CHECK-SAME:         outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:     flow.dispatch.tensor.store %[[RESULT]], %[[ARG6]], {{.*}}

// -----

func @fuse_matmul_with_fill(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = constant 0.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = memref.dim %A, %c0 : tensor<?x?xf32>
  %N = memref.dim %B, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%M, %N] : tensor<?x?xf32>
  %1 = linalg.fill(%0, %zero) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %2 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//       CHECK:   func @fuse_matmul_with_fill
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//       CHECK:     %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//       CHECK:     %[[N:.+]] = memref.dim %[[ARG1]], %[[C1]]
//       CHECK:     flow.dispatch.workgroups[%[[N]], %[[M]], %[[C1]]]
//  CHECK-SAME:       (%[[M]], %[[N]], %[[ARG0]], %[[ARG1]])
//  CHECK-NEXT:       (%[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:        %[[ARG3:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:        %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  CHECK-SAME:        %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  CHECK-SAME:        %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>) {
//       CHECK:        %[[ZERO:.+]] = constant 0.000000e+00 : f32
//       CHECK:        scf.for
//       CHECK:          scf.for
//   CHECK-DAG:            %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[ARG4]], {{.*}}
//   CHECK-DAG:            %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//   CHECK-DAG:            %[[INIT_TILE:.+]] = linalg.init_tensor
//       CHECK:            %[[FILL_TILE:.+]] = linalg.fill(%[[INIT_TILE]], %[[ZERO]])
//       CHECK:            %[[RESULT_TILE:.+]] = linalg.matmul
//  CHECK-SAME:              ins(%[[LHS_TILE]], %[[RHS_TILE]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:              outs(%[[FILL_TILE]] : tensor<?x?xf32>)
//       CHECK:            flow.dispatch.tensor.store %[[RESULT_TILE]], %[[ARG6]], {{.*}}
//       CHECK:          flow.return
//       CHECK:        }

// -----

func @keep_separate_dispatches_for_producer(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = constant 0.0 : f32
  %one = constant 1.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = memref.dim %A, %c0 : tensor<?x?xf32>
  %N = memref.dim %B, %c1 : tensor<?x?xf32>
  %K = memref.dim %A, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%M, %N] : tensor<?x?xf32>
  %1 = linalg.fill(%0, %zero) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %2 = linalg.init_tensor [%M, %K] : tensor<?x?xf32>
  %3 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%A : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
      %4 = addf %arg0, %one : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%3, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
//      CHECK: func @keep_separate_dispatches_for_producer
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[M:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:     %[[N:.+]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:     %[[K:.+]] = memref.dim %[[ARG0]], %[[C1]]
//       CHECK:     %[[RESULT1:.+]] = flow.dispatch.workgroups[%[[K]], %[[M]], %[[C1]]]
//  CHECK-SAME:       (%[[ARG0]], %[[M]], %[[K]])
//  CHECK-NEXT:       (%[[ARG2:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  CHECK-SAME:        %[[ARG3:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:        %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:        %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>) {
//       CHECK:          %[[ONE:.+]] = constant 1.0
//   CHECK-DAG:          %[[INPUT:.+]] = flow.dispatch.tensor.load %[[ARG2]]
//   CHECK-DAG:          %[[INIT:.+]] = linalg.init_tensor
//       CHECK:          %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:            ins(%[[INPUT]] : tensor<?x?xf32>)
//  CHECK-SAME:            outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:          flow.dispatch.tensor.store %[[RESULT]], %[[ARG5]]
//       CHECK:          flow.return
//       CHECK:     }
//       CHECK:     flow.dispatch.workgroups[%[[N]], %[[M]], %[[C1]]]
//       CHECK:       %[[ZERO:.+]] = constant 0.0
//       CHECK:       scf.for
//       CHECK:         scf.for
//       CHECK:            %[[INIT_TILE:.+]] = linalg.init_tensor
//       CHECK:            %[[FILL_TILE:.+]] = linalg.fill(%[[INIT_TILE]], %[[ZERO]])
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
//       NOCHECK:          %[[ZERO:.+]] = constant 0.0
//       NOCHECK:          scf.for
//       NOCHECK:            scf.for
//   NOCHECK-DAG:              %[[LHS_TILE_2:.+]] = flow.dispatch.tensor.load %[[ARG6]], {{.*}}
//   NOCHECK-DAG:              %[[RHS_TILE_2:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//   NOCHECK-DAG:              %[[INIT_TILE_2:.+]] = linalg.init_tensor
//       NOCHECK:              %[[FILL_TILE:.+]] = linalg.fill(%[[INIT_TILE]], %[[ZERO]])
//       NOCHECK:              %[[RESULT_TILE_2:.++]]] = linalg.matmul
//  NOCHECK-SAME:                ins(%[[LHS_TILE_2]], %[[RHS_TILE_2]] : tensor<?x?xf32>, tensor<?x?xf32>)
//       NOCHECK:                outs(%[[FILL_TILE_2]] : tensor<?x?xf32>)
//       NOCHECK:              flow.dispatch.tensor.store %[[RESULT_TILE_2]], %[[ARG7]]
//       NOCHECK:          flow.return
//       NOCHECK:        }

// -----

func @fuse_reshape_op(%arg0: tensor<?x?xf32>) -> tensor<?xf32>
{
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  return %0 : tensor<?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: func @fuse_reshape_op
// CHECK-SAME:   (%[[ARG0:.+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[WORKLOAD:.+]] = affine.apply #[[MAP0]]()[%[[D0]], %[[D1]]]
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
// CHECK-SAME:     [%[[WORKLOAD]], %[[C1]], %[[C1]]](%[[ARG0]])
// CHECK-NEXT:     %[[ARG1:.+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG2:.+]]: !flow.dispatch.tensor<writeonly:?xf32>
//      CHECK:       %[[LOAD:.+]] = flow.dispatch.tensor.load %[[ARG1]], {{.*}}
//      CHECK:       %[[RESHAPE:.+]] = linalg.tensor_collapse_shape %[[LOAD]] {{\[}}[0, 1]]
//      CHECK:       flow.dispatch.tensor.store %[[RESHAPE]], %[[ARG2]], {{.*}}

// -----

func @tile_4d_generic_op_alone
  (%A: tensor<?x?x?x?xf32>, %B: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %d0 = memref.dim %A, %c0 : tensor<?x?x?x?xf32>
  %d1 = memref.dim %A, %c1 : tensor<?x?x?x?xf32>
  %d2 = memref.dim %A, %c2 : tensor<?x?x?x?xf32>
  %d3 = memref.dim %A, %c3 : tensor<?x?x?x?xf32>
  %0 = linalg.init_tensor [%d0, %d1, %d2, %d3] : tensor<?x?x?x?xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins (%A, %B: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%0 : tensor<?x?x?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %2 = addf %arg0, %arg1 : f32
        linalg.yield %2 : f32
    } -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
// For ops of rank greater than 3 we serialized the higher dimension. When flow
// supports larger ranks this can be changed.
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: func @tile_4d_generic_op_alone
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = constant 3 : index
//  CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = memref.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[D3:.+]] = memref.dim %[[ARG0]], %[[C3]]
//  CHECK-DAG:   %[[WG_SISE_2:.+]] = flow.dispatch.workgroup.size[2] : index
//  CHECK-DAG:   %[[WG_ID_2:.+]] = flow.dispatch.workgroup.id[2] : index
//  CHECK-DAG:   flow.dispatch.workgroups[%[[D3]], %[[D2]], %[[D1]]]
//  CHECK-DAG:   %[[D4:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_2]], %[[WG_SISE_2]]]

// -----

func @always_fuse_reshape
  (%lhs : tensor<?xf32>, %rhs1 : tensor<4x?xf32>, %rhs2 : tensor<4x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %cst = constant 0.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = linalg.tensor_expand_shape %lhs [[0, 1]]
    : tensor<?xf32> into tensor<?x4xf32>
  %m = memref.dim %0, %c0 : tensor<?x4xf32>
  %n1 = memref.dim %rhs1, %c1 : tensor<4x?xf32>
  %init1 = linalg.init_tensor [%m, %n1] : tensor<?x?xf32>
  %fill1 = linalg.fill(%init1, %cst) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %1 = linalg.matmul
    ins(%0, %rhs1 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %n2 = memref.dim %rhs2, %c1 : tensor<4x?xf32>
  %init2 = linalg.init_tensor [%m, %n2] : tensor<?x?xf32>
  %fill2 = linalg.fill(%init2, %cst) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
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
//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//      CHECK:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[M:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//  CHECK-DAG:   %[[N1:.+]] = memref.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RESULT1:.+]] = flow.dispatch.workgroups[%[[N1]], %[[M]], %[[C1]]]
// CHECK-SAME:     (%[[M]], %[[N1]], %[[ARG0]], %[[RHS1]])
//      CHECK:   %[[N2:.+]] = memref.dim %[[RHS2]], %[[C1]]
//      CHECK:   %[[RESULT2:.+]] = flow.dispatch.workgroups[%[[N2]], %[[M]], %[[C1]]]
// CHECK-SAME:     (%[[M]], %[[N2]], %[[ARG0]], %[[RHS2]])
//      CHECK:   return %[[RESULT1]], %[[RESULT2]]

// -----

func @fuse_tensor_update_with_fill(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: index,
               %arg3: index, %arg4: index, %arg5: index) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %2 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%1)[%arg2, %arg4]
  %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%2)[%arg3, %arg5]
  %5 = linalg.init_tensor [%3, %4] : tensor<?x?xf32>
  %6 = linalg.fill(%5, %0) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %7 = flow.tensor.update %arg0, %6[%arg2, %arg3] : tensor<?x?xf32>{%1, %2} -> tensor<?x?xf32>{%3, %4}
  return %7 : tensor<?x?xf32>
}

//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
//       CHECK: func @fuse_tensor_update_with_fill
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[RD0:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]], %[[D0]]]
//   CHECK-DAG:   %[[RD1:.+]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG5]], %[[D1]]]
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:    [%[[RD1]], %[[RD0]], %[[C1]]]
//  CHECK-SAME:    (%[[ARG1]], %[[RD0]], %[[RD1]])
//   CHECK-DAG:      %[[VAL:.+]] = tensor.extract
//   CHECK-DAG:      %[[INIT:.+]] = linalg.init_tensor
//       CHECK:      %[[RETURN:.+]] = linalg.fill(%[[INIT]], %[[VAL]])
//       CHECK:      flow.dispatch.tensor.store %[[RETURN]], {{.*}}
//  CHECK-NEXT:      flow.return
//       CHECK:   flow.tensor.update %[[ARG0]], %[[RESULT]]

// -----

// CHECK-LABEL: func @pass_constant_through()
func @pass_constant_through() -> tensor<2x2x3xi32> {
  // CHECK: %[[CST:.+]] = constant dense<{{.+}}> : tensor<2x2x3xi32>
  %cst = constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  // CHECK: return %[[CST]]
  return %cst : tensor<2x2x3xi32>
}

// -----

// CHECK-LABEL: func @fuse_matmul_with_generic_op
func @fuse_matmul_with_generic_op(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> {
  %f12 = constant 12.0 : f32

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
  %f12 = constant 12.0 : f32
  //  CHECK-DAG: %[[C0:.+]] = constant 0 : index
  //  CHECK-DAG: %[[C1:.+]] = constant 1 : index
  //  CHECK-DAG: %[[D0:.+]] = memref.dim %[[ARG2]], %[[C0]]
  //  CHECK-DAG: %[[D1:.+]] = memref.dim %[[ARG2]], %[[C1]]
  //      CHECK: %[[origCC:.+]] = flow.dispatch.workgroups[%[[D1]], %[[D0]], %[[C1]]](%[[ARG2]])
  // CHECK-NEXT:   %[[ARG3:.+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
  //      CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[ARG3]], {{.*}}
  //      CHECK:   %[[STOREVAL:.+]] = linalg.generic
  // CHECK-SAME:     outs(%[[LOAD]] : tensor<?x?xf32>)
  //      CHECK:   flow.dispatch.tensor.store %[[STOREVAL]], %[[ARG3]], {{.*}}

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
  %cst = constant 0.000000e+00 : f32
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  return %2 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func @conv2d
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.conv_2d_input_nhwc_filter_hwcf

// -----

func @depthwise_conv2d(%input: tensor<1x113x113x96xf32>, %filter: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
  %cst = constant 0.000000e+00 : f32
  %1 = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
  %2 = linalg.fill(%1, %cst) : tensor<1x56x56x96xf32>, f32 -> tensor<1x56x56x96xf32>
  %4 = linalg.depthwise_conv_2d_input_nhwc_filter_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%input, %filter : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%2 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  return %4 : tensor<1x56x56x96xf32>
}

// CHECK-LABEL: func @depthwise_conv2d
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.depthwise_conv_2d_input_nhwc_filter_hwc

// -----

func @subtensor_insert(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x225x225x3xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 225, 225, 3] : tensor<1x225x225x3xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x225x225x3xf32>, f32 -> tensor<1x225x225x3xf32>
  %2 = subtensor_insert %arg0 into %1[0, 0, 0, 0] [1, 224, 224, 3] [1, 1, 1, 1] : tensor<1x224x224x3xf32> into tensor<1x225x225x3xf32>
  return %2 : tensor<1x225x225x3xf32>
}

//      CHECK: func @subtensor_insert
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x224x224x3xf32>)
//
//      CHECK:   %[[FILL:.+]] = flow.dispatch.workgroups[{{.+}}]() : () -> tensor<1x225x225x3xf32> =
// CHECK-NEXT:       (%[[OUTPUT:.+]]: !flow.dispatch.tensor<writeonly:1x225x225x3xf32>) {
//      CHECK:     linalg.init_tensor
// CHECK-NEXT:     %[[TENSOR:.+]] = linalg.fill
// CHECK-NEXT:     flow.dispatch.tensor.store %[[TENSOR]], %[[OUTPUT]], {{.*}}
// CHECK-NEXT:     flow.return
//
//      CHECK:   %[[PAD:.+]] = flow.dispatch.workgroups[{{.+}}](%[[INPUT]], %[[FILL]]) : (tensor<1x224x224x3xf32>, tensor<1x225x225x3xf32>) -> %[[FILL]] =
// CHECK-NEXT:       (%[[SRC:.+]]: !flow.dispatch.tensor<readonly:1x224x224x3xf32>, %[[DST:.+]]: !flow.dispatch.tensor<readwrite:1x225x225x3xf32>) {
// CHECK-NEXT:     %[[SRC_TENSOR:.+]] = flow.dispatch.tensor.load %[[SRC]], {{.*}} : !flow.dispatch.tensor<readonly:1x224x224x3xf32> -> tensor<1x224x224x3xf32>
// CHECK-NEXT:     %[[DST_TENSOR:.+]] = flow.dispatch.tensor.load %[[DST]], {{.*}} : !flow.dispatch.tensor<readwrite:1x225x225x3xf32> -> tensor<1x225x225x3xf32>
// CHECK-NEXT:     %[[INSERT:.+]] = subtensor_insert %[[SRC_TENSOR]] into %[[DST_TENSOR]][0, 0, 0, 0] [1, 224, 224, 3] [1, 1, 1, 1]
// CHECK-NEXT:     flow.dispatch.tensor.store %[[INSERT]], %[[DST]], {{.*}} : tensor<1x225x225x3xf32> -> !flow.dispatch.tensor<readwrite:1x225x225x3xf32>
// CHECK-NEXT:     flow.return
//
//      CHECK:   return %[[PAD]] : tensor<1x225x225x3xf32>

// -----

func @fuse_non_tiled_reduction_fill(%input1: tensor<1000xf32>, %input2: tensor<1000xf32>, %offset: tensor<f32>) -> tensor<f32> {
  %zero = constant 0.0 : f32
  %init = linalg.init_tensor [] : tensor<f32>
  %fill = linalg.fill(%init, %zero) : tensor<f32>, f32 -> tensor<f32>
  %reduce = linalg.generic {
              indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
              iterator_types = ["reduction"]}
            ins(%input1, %input2, %offset : tensor<1000xf32>, tensor<1000xf32>, tensor<f32>)
            outs(%fill : tensor<f32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    %555 = addf %arg1, %arg2 : f32
    %556 = subf %555, %arg3 : f32
    %557 = math.exp %556 : f32
    %558 = addf %557, %arg4 : f32
    linalg.yield %558 : f32
  } -> tensor<f32>
  return %reduce : tensor<f32>
}

// CHECK-LABEL: func @fuse_non_tiled_reduction_fill

//      CHECK: %[[C1:.+]] = constant 1 : index
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
  %0 = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
  %1 = subtensor %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = linalg.tensor_collapse_shape %1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %3 = linalg.tensor_collapse_shape %arg1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %4 = subtensor %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = linalg.tensor_collapse_shape %4 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %6 = subtensor %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = linalg.tensor_collapse_shape %6 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %8 = linalg.init_tensor [%arg3] : tensor<?xf32>
  %9 = linalg.generic {
      indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map1],
      iterator_types = ["parallel"]}
      ins(%arg2, %2, %3, %5, %7 : tensor<i32>, tensor<?xf32>,
          tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      outs(%8 : tensor<?xf32>) {
      ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32):
        %10 = addf %arg5, %arg6 : f32
        %11 = addf %arg7, %arg8 : f32
        %12 = addf %10, %11 : f32
        %13 = sitofp %arg4 : i32 to f32
        %14 = addf %12, %13 : f32
        linalg.yield %14 : f32
      } -> tensor<?xf32>
  return %9 : tensor<?xf32>
}
// CHECK-LABEL: func @inline_dag_1
//   CHECK-NOT:   linalg.
//   CHECK-NOT:   subtensor
//       CHECK:   flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x?xf32>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?xf32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xf32>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG6]], {{.*}}
//       CHECK:     %[[OP1:.+]] = linalg.tensor_expand_shape %[[LEAF2]]
//       CHECK:     %[[OP2:.+]] = linalg.tensor_collapse_shape %[[LEAF1]]
//       CHECK:     %[[OP3:.+]] = subtensor %[[OP1]][0, 0]
//       CHECK:     %[[OP4:.+]] = subtensor %[[OP1]][0, 10]
//       CHECK:     %[[OP5:.+]] = subtensor %[[OP1]][0, 20]
//       CHECK:     %[[OP6:.+]] = linalg.tensor_collapse_shape %[[OP3]]
//       CHECK:     %[[OP7:.+]] = linalg.tensor_collapse_shape %[[OP4]]
//       CHECK:     %[[OP8:.+]] = linalg.tensor_collapse_shape %[[OP5]]

// -----

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
func @inline_dag_2(
    %arg0: tensor<?xf32>, %arg1 : tensor<1x?xf32>, %arg2 : tensor<i32>,
    %arg3 : index) -> tensor<?xf32> {
  %0 = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<1x?xf32>
  %1 = subtensor %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = linalg.tensor_collapse_shape %arg1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  br ^bb1
^bb1:
  %3 = linalg.tensor_collapse_shape %1 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %4 = subtensor %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = linalg.tensor_collapse_shape %4 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %6 = subtensor %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = linalg.tensor_collapse_shape %6 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  %8 = linalg.init_tensor [%arg3] : tensor<?xf32>
  %9 = linalg.generic {
      indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map1],
      iterator_types = ["parallel"]}
      ins(%arg2, %3, %2, %5, %7 : tensor<i32>, tensor<?xf32>,
          tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      outs(%8 : tensor<?xf32>) {
      ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32):
        %10 = addf %arg5, %arg6 : f32
        %11 = addf %arg7, %arg8 : f32
        %12 = addf %10, %11 : f32
        %13 = sitofp %arg4 : i32 to f32
        %14 = addf %12, %13 : f32
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
//       CHECK:     %[[OP1:.+]] = linalg.tensor_expand_shape %[[LEAF2]]
//       CHECK:     %[[OP2:.+]] = linalg.tensor_collapse_shape %[[LEAF1]]
//       CHECK:     %[[OP3:.+]] = subtensor %[[OP1]][0, 0]
//       CHECK:     %[[OP4:.+]] = subtensor %[[OP1]][0, 10]
//       CHECK:     %[[OP5:.+]] = subtensor %[[OP1]][0, 20]
//       CHECK:     %[[OP6:.+]] = linalg.tensor_collapse_shape %[[OP3]]
//       CHECK:     %[[OP7:.+]] = linalg.tensor_collapse_shape %[[OP4]]
//       CHECK:     %[[OP8:.+]] = linalg.tensor_collapse_shape %[[OP5]]

// -----

func @inline_dag_3(%240 : tensor<9xi32>, %244 : tensor<18xi32>, %247 : tensor<i32>) -> tensor<9xi1> {
  %c9 = constant 9 : index
  %c5_i32 = constant 5 : i32
  %c0_i32 = constant 0 : i32
  %c9_i32 = constant 9 : i32
  %245 = flow.tensor.update %240, %244[%c9] : tensor<9xi32> -> tensor<18xi32>
  %248 = tensor.extract %247[] : tensor<i32>
  %249 = cmpi slt, %248, %c9_i32 : i32
  %250 = select %249, %248, %c9_i32 : i32
  %251 = cmpi sgt, %250, %c0_i32 : i32
  %252 = select %251, %250, %c0_i32 : i32
  %253 = index_cast %252 : i32 to index
  %254 = subtensor %245[%253] [9] [1] : tensor<18xi32> to tensor<9xi32>
  %255 = linalg.init_tensor [9] : tensor<9xi1>
  %256 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%254 : tensor<9xi32>) outs(%255 : tensor<9xi1>) {
        ^bb0(%arg20: i32, %arg21: i1):  // no predecessors
          %849 = cmpi eq, %arg20, %c5_i32 : i32
          linalg.yield %849 : i1
      } -> tensor<9xi1>
  return %256 : tensor<9xi1>
}
// CHECK-LABEL: func @inline_dag_3
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<9xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<18xi32>
//  CHECK-SAME:   %[[ARG2:.+]]: tensor<i32>
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.update %[[ARG0]], %[[ARG1]]
//       CHECK:   flow.dispatch.workgroups
//  CHECK-SAME:     (%[[UPDATE]], %[[ARG2]])
//  CHECK-NEXT:     (%[[ARG3:.+]]: !flow.dispatch.tensor<readonly:18xi32>
//  CHECK-SAME:      %[[ARG4:.+]]: !flow.dispatch.tensor<readonly:i32>,
//   CHECK-DAG:     %[[C5:.+]] = constant 5 : i32
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : i32
//   CHECK-DAG:     %[[C9:.+]] = constant 9 : i32
//   CHECK-DAG:     %[[ARG3V:.+]] = flow.dispatch.tensor.load %[[ARG3]]
//   CHECK-DAG:     %[[ARG4V:.+]] = flow.dispatch.tensor.load %[[ARG4]]
//   CHECK-DAG:     %[[EXTRACT:.+]] = tensor.extract %[[ARG4V]]
//   CHECK-DAG:     %[[CMP1:.+]] = cmpi slt, %[[EXTRACT]]
//   CHECK-DAG:     %[[SELECT1:.+]] = select %[[CMP1]], %[[EXTRACT]], %[[C9]]
//   CHECK-DAG:     %[[CMP2:.+]] = cmpi sgt, %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[SELECT2:.+]] = select %[[CMP2]], %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[INDEX_CAST:.+]] = index_cast %[[SELECT2]]
//   CHECK-DAG:     subtensor %[[ARG3V]][%[[INDEX_CAST]]]
//       CHECK:     flow.return

// -----

#map = affine_map<() -> ()>
func @inline_dag_4(%arg0: tensor<4xi32>, %arg1: tensor<i32>) -> tensor<i16> {
  %c3_i32 = constant 3 : i32
  %c0_i32 = constant 0 : i32
  %0 = tensor.extract %arg1[] : tensor<i32>
  %1 = cmpi slt, %0, %c3_i32 : i32
  %2 = select %1, %0, %c3_i32 : i32
  %3 = cmpi sgt, %2, %c0_i32 : i32
  %4 = select %3, %2, %c0_i32 : i32
  %5 = index_cast %4 : i32 to index
  %6 = subtensor %arg0[%5] [1] [1] : tensor<4xi32> to tensor<i32>
  br ^bb1
^bb1:  // pred: ^bb0
  %7 = linalg.init_tensor [] : tensor<i16>
  %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%6 : tensor<i32>) outs(%7 : tensor<i16>) {
  ^bb0(%arg2: i32, %arg3: i16):  // no predecessors
    %9 = trunci %arg2 : i32 to i16
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
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : i32
//   CHECK-DAG:     %[[C3:.+]] = constant 3 : i32
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG2]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG3]], {{.*}}
//       CHECK:     %[[INIT:.+]] = linalg.init_tensor [] : tensor<i16>
//       CHECK:     %[[OP1:.+]] = tensor.extract %[[LEAF2]][] : tensor<i32>
//       CHECK:     %[[OP2:.+]] = cmpi slt, %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP3:.+]] = select %[[OP2]], %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP4:.+]] = cmpi sgt, %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP5:.+]] = select %[[OP4]], %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP6:.+]] = index_cast %[[OP5]] : i32 to index
//       CHECK:     %[[OP7:.+]] = subtensor %[[LEAF1]][%[[OP6]]] [1] [1] : tensor<4xi32> to tensor<i32>
//       CHECK:     %[[RES:.+]] = linalg.generi
//  CHECK-SAME:       ins(%[[OP7]] : tensor<i32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<i16>) {
//       CHECK:     ^bb0(%[[ARG5:.+]]: i32, %{{.+}}: i16):  // no predecessors
//       CHECK:       %[[TRUNC:.+]] = trunci %[[ARG5]] : i32 to i16
//       CHECK:       linalg.yield %[[TRUNC]] : i16
//       CHECK:     } -> tensor<i16>
//       CHECK:     flow.dispatch.tensor.store %[[RES]], %[[ARG4]]

// -----

func @multi_result(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
  %cmin = constant -2147483648 : i32
  %c0_i32 = constant 0 : i32
  %c0 = constant 0 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = linalg.init_tensor [%0] : tensor<?xi32>
  %2 = linalg.fill(%1, %cmin) : tensor<?xi32>, i32 -> tensor<?xi32>
  %3 = linalg.fill(%1, %c0_i32) : tensor<?xi32>, i32 -> tensor<?xi32>
  %4:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
      outs(%1, %2 : tensor<?xi32>, tensor<?xi32>) {
  ^bb0(%arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
    %5 = cmpi sge, %arg2, %arg4 : i32
    %6 = select %5, %arg2, %arg4 : i32
    %7 = cmpi eq, %arg2, %arg4 : i32
    %8 = cmpi slt, %arg3, %arg5 : i32
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

func @multi_result_fallback(%arg0: tensor<?x10xi32>, %arg1: tensor<?x10xi32>)
    -> (tensor<?x10xi32>, tensor<?x10xi32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x10xi32>
  %1 = linalg.init_tensor [%0, 10] : tensor<?x10xi32>
  %2:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, 10-d1)>,
                       affine_map<(d0, d1) -> (d0, 10-d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x10xi32>, tensor<?x10xi32>)
      outs(%1, %1 : tensor<?x10xi32>, tensor<?x10xi32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32):
        linalg.yield %arg2, %arg3 : i32, i32
      } -> (tensor<?x10xi32>, tensor<?x10xi32>)
  return %2#0, %2#1 : tensor<?x10xi32>, tensor<?x10xi32>
}
// CHECK-LABEL: func @multi_result_fallback
//       CHECK:   %[[RESULT:.+]]:2 = flow.dispatch.workgroup
//  CHECK-NEXT:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x10xi32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x10xi32>
//   CHECK-NOT:     scf.for
//       CHECK:     %[[OP_RESULT:.+]]:2 = linalg.generic
//   CHECK-DAG:     flow.dispatch.tensor.store %[[OP_RESULT]]#0, %[[ARG5]]
//   CHECK-DAG:     flow.dispatch.tensor.store %[[OP_RESULT]]#1, %[[ARG6]]
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func @dynamic_slice(%arg0: tensor<?x?xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3 : index) -> tensor<1x?xi32> {
  %c1_i32 = constant 1 : i32
  %c0_i32 = constant 0 : i32
  %0 = tensor.extract %arg1[] : tensor<i32>
  %1 = cmpi slt, %0, %c1_i32 : i32
  %2 = select %1, %0, %c1_i32 : i32
  %3 = cmpi sgt, %2, %c0_i32 : i32
  %4 = select %3, %2, %c0_i32 : i32
  %5 = index_cast %4 : i32 to index
  %6 = tensor.extract %arg2[] : tensor<i32>
  %7 = cmpi slt, %6, %c0_i32 : i32
  %8 = select %7, %6, %c0_i32 : i32
  %9 = cmpi sgt, %8, %c0_i32 : i32
  %10 = select %9, %8, %c0_i32 : i32
  %11 = index_cast %10 : i32 to index
  %12 = subtensor %arg0[%5, %11] [1, %arg3] [1, 1] : tensor<?x?xi32> to tensor<1x?xi32>
  return %12 : tensor<1x?xi32>
}
// CHECK-LABEL: func @dynamic_slice(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<i32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<i32>
//  CHECK-SAME:   %[[ARG3:.+]]: index
//       CHECK:   %[[C1:.+]] = constant 1 : index
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:     [%[[ARG3]], %[[C1]], %[[C1]]]
//  CHECK-SAME:     (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
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
//       CHECK:     subtensor
//       CHECK:     flow.return
//       CHECK:   return %[[RESULT]]

// -----

func @dynamic_dot() -> !hal.buffer_view attributes {iree.abi.stub} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cst = constant 0.000000e+00 : f32
  %0 = iree.dynamic_shape_constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01], [1.200000e+01, 1.100000e+01, 1.000000e+01], [9.000000e+00, 8.000000e+00, 7.000000e+00], [6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<5x3xf32> -> tensor<?x?xf32>
  %1 = iree.dynamic_shape_constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01, 1.200000e+01, 1.100000e+01], [1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<3x5xf32> -> tensor<?x?xf32>
  %2 = memref.dim %0, %c0 : tensor<?x?xf32>
  %3 = memref.dim %1, %c1 : tensor<?x?xf32>
  %4 = linalg.init_tensor [%2, %3] : tensor<?x?xf32>
  %5 = linalg.fill(%4, %cst) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %6 = linalg.matmul ins(%0, %1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = memref.dim %6, %c0 : tensor<?x?xf32>
  %8 = memref.dim %6, %c1 : tensor<?x?xf32>
  %9 = hal.tensor.cast %6 : tensor<?x?xf32>{%7, %8} -> !hal.buffer_view
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