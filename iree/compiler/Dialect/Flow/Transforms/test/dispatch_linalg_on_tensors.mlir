// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-dispatch-linalg-on-tensors-pass -canonicalize -cse %s | IreeFileCheck %s

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
// CHECK: #[[MULMAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
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
//  CHECK-DAG:         %[[LOAD2:.+]] = flow.dispatch.tensor.load %[[ARG2]], {{.*}}
//  CHECK-DAG:         %[[INIT:.+]] = linalg.init_tensor
//  CHECK-DAG:         %[[LOAD3:.+]] = flow.dispatch.tensor.load %[[ARG3]], {{.*}}
//      CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:           ins(%[[LOAD2]], %[[LOAD3]] : tensor<?x?xf32>, tensor<?xf32>)
// CHECK-SAME:           outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:         flow.dispatch.tensor.store %[[RESULT]], %[[ARG6]], {{.*}}

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
//   CHECK-DAG:          %[[INPUT:.+]] = flow.dispatch.tensor.load %[[ARG2]], {{.*}}
//   CHECK-DAG:          %[[INIT:.+]] = linalg.init_tensor
//       CHECK:          %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:            ins(%[[INPUT]] : tensor<?x?xf32>)
//  CHECK-SAME:            outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:          flow.dispatch.tensor.store %[[RESULT]], %[[ARG5]], {{,*}}
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
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<?x?xf32> into tensor<?xf32>
  return %0 : tensor<?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
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
//      CHECK:       %[[RESHAPE:.+]] = linalg.tensor_reshape %[[LOAD]] [#[[MAP1]]]
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
//      CHECK:   flow.dispatch.workgroups[%[[D3]], %[[D2]], %[[D1]]]

// -----

func @always_fuse_reshape
  (%lhs : tensor<?xf32>, %rhs1 : tensor<4x?xf32>, %rhs2 : tensor<4x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %cst = constant 0.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = linalg.tensor_reshape %lhs [affine_map<(d0, d1) -> (d0, d1)>]
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
  -> tensor<?x?xf32> attributes {iree.module.export}
{
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
  -> (tensor<?x?xf32>, tensor<?x?xf32>) attributes {iree.module.export}
{
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
  %4 = linalg.depthwise_conv_2d_input_nhwc_filter_hwc {strides = dense<2> : tensor<2xi64>} ins(%input, %filter : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%2 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
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

func @tile_parallel_reduction(%arg0: tensor<7x7x1280xf32>) -> tensor<1280xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1280] : tensor<1280xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1280xf32>, f32 -> tensor<1280xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg0 : tensor<7x7x1280xf32>) outs(%1 : tensor<1280xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %3 = addf %arg1, %arg2 : f32
    linalg.yield %3 : f32
  } -> tensor<1280xf32>
  return %2 : tensor<1280xf32>
}

//  CHECK-DAG: #[[SIZE_MAP0:.+]] = affine_map<(d0, d1) -> (d1, -d0 + 1280)>
//  CHECK-DAG: #[[SIZE_MAP1:.+]] = affine_map<(d0, d1) -> (-d0 + 1280, d1)>

//      CHECK: func @tile_parallel_reduction
// CHECK-SAME: (%[[INPUT:.+]]: tensor<7x7x1280xf32>)

//  CHECK-DAG: %[[C1:.+]] = constant 1 : index
//  CHECK-DAG: %[[C1280:.+]] = constant 1280 : index
//      CHECK: %[[REDUCE:.+]] = flow.dispatch.workgroups[%[[C1280]], %[[C1]], %[[C1]]](%[[INPUT]]) : (tensor<7x7x1280xf32>) -> tensor<1280xf32> =
// CHECK-NEXT:     (%[[ARG1:.+]]: !flow.dispatch.tensor<readonly:7x7x1280xf32>, %[[ARG2:.+]]: !flow.dispatch.tensor<writeonly:1280xf32>) {
//      CHECK:   %[[WG_SIZE0:.+]] = flow.dispatch.workgroup.size[0] : index
//      CHECK:   scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
//      CHECK:     %[[SIZE0:.+]] = affine.min #[[SIZE_MAP0]](%[[IV]], %[[WG_SIZE0]])
//      CHECK:     %[[IN:.+]] = flow.dispatch.tensor.load %[[ARG1]],
// CHECK-SAME:       sizes = [7, 7, %[[SIZE0]]]
//      CHECK:     %[[SIZE1:.+]] = affine.min #[[SIZE_MAP1]](%[[IV]], %[[WG_SIZE0]])
//      CHECK:     %[[INIT:.+]] = linalg.init_tensor [%[[SIZE1]]] : tensor<?xf32>
// CHECK-NEXT:     %[[OUT:.+]] = linalg.fill(%[[INIT]]
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[IN]] : tensor<7x7x?xf32>) outs(%[[OUT]] : tensor<?xf32>)
//      CHECK:     flow.dispatch.tensor.store %[[GENERIC]], %[[ARG2]], {{.*}}

//      CHECK: return %[[REDUCE]]

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
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<?xf32> into tensor<1x?xf32>
  %1 = subtensor %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = linalg.tensor_reshape %1 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
  %3 = linalg.tensor_reshape %arg1 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
  %4 = subtensor %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = linalg.tensor_reshape %4 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
  %6 = subtensor %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = linalg.tensor_reshape %6 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
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
//       CHECK:     %[[OP1:.+]] = linalg.tensor_reshape %[[LEAF2]]
//       CHECK:     %[[OP2:.+]] = linalg.tensor_reshape %[[LEAF1]]
//       CHECK:     %[[OP3:.+]] = subtensor %[[OP1]][0, 0]
//       CHECK:     %[[OP4:.+]] = subtensor %[[OP1]][0, 10]
//       CHECK:     %[[OP5:.+]] = subtensor %[[OP1]][0, 20]
//       CHECK:     %[[OP6:.+]] = linalg.tensor_reshape %[[OP3]]
//       CHECK:     %[[OP7:.+]] = linalg.tensor_reshape %[[OP4]]
//       CHECK:     %[[OP8:.+]] = linalg.tensor_reshape %[[OP5]]

// -----

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
func @inline_dag_2(
    %arg0: tensor<?xf32>, %arg1 : tensor<1x?xf32>, %arg2 : tensor<i32>,
    %arg3 : index) -> tensor<?xf32> {
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<?xf32> into tensor<1x?xf32>
  %1 = subtensor %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = linalg.tensor_reshape %arg1 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
  br ^bb1
^bb1:
  %3 = linalg.tensor_reshape %1 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
  %4 = subtensor %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = linalg.tensor_reshape %4 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
  %6 = subtensor %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = linalg.tensor_reshape %6 [affine_map<(d0, d1) -> (d0, d1)>] : tensor<1x?xf32> into tensor<?xf32>
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
//       CHECK:   %[[OP1:.+]] = linalg.tensor_reshape %[[ARG0]]
//       CHECK:   subtensor %[[OP1]]
//       CHECK:   linalg.tensor_reshape %[[ARG1]]
//       CHECK:   flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x?xf32>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?xf32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x?xf32>
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xf32>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG6]], {{.*}}
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG7]], {{.*}}
//       CHECK:     %[[OP1:.+]] = subtensor %[[LEAF2]][0, 0]
//       CHECK:     %[[OP2:.+]] = subtensor %[[LEAF2]][0, 10]
//       CHECK:     %[[OP3:.+]] = linalg.tensor_reshape %[[LEAF1]]
//       CHECK:     %[[OP4:.+]] = linalg.tensor_reshape %[[OP1]]
//       CHECK:     %[[OP5:.+]] = linalg.tensor_reshape %[[OP2]]

// -----

func @fuse_conv2d_elementwise(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>, %offset: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  %3 = linalg.generic {
         indexing_maps = [
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
           affine_map<(d0, d1, d2, d3) -> (d3)>,
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
         ins(%2, %offset: tensor<1x112x112x32xf32>, tensor<32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>) {
         ^bb0(%a: f32, %b: f32, %c: f32):
            %sub = subf %a, %b : f32
            linalg.yield %sub : f32
         } -> tensor<1x112x112x32xf32>
  return %3 : tensor<1x112x112x32xf32>
}

// Check that
// * linalg.conv is fused together with linalg.generic;
// * linalg.generic's linalg.fill is pulled into the same group;
// * linalg.conv's linalg.fill is pulled into the same group.

// CHECK-LABEL: func @fuse_conv2d_elementwise

//      CHECK: flow.dispatch.workgroups
//      CHECK:   scf.for
//      CHECK:     scf.for
//      CHECK:       %[[INIT:.+]] = linalg.init_tensor
//      CHECK:       %[[FILL0:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:       %[[CONV:.+]] = linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:         outs(%[[FILL0]] : tensor<1x?x?x?xf32>)
//      CHECK:       %[[FILL1:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:       linalg.generic
// CHECK-SAME:         ins(%[[CONV]], %{{.+}} : tensor<1x?x?x?xf32>, tensor<?xf32>)
// CHECK-SAME:         outs(%[[FILL1]] : tensor<1x?x?x?xf32>)

// -----

func @dont_fuse_conv2d_with_multiple_uses(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>, %offset: tensor<32xf32>)
  -> (tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  %3 = linalg.generic {
         indexing_maps = [
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
           affine_map<(d0, d1, d2, d3) -> (d3)>,
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
         ins(%2, %offset: tensor<1x112x112x32xf32>, tensor<32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>) {
         ^bb0(%a: f32, %b: f32, %c: f32):
            %sub = subf %a, %b : f32
            linalg.yield %sub : f32
         } -> tensor<1x112x112x32xf32>
  return %3, %2 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>
}

// CHECK-LABLE: func @dont_fuse_conv2d_with_multiple_uses

// CHECK: flow.dispatch.workgroups
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.conv_2d_input_nhwc_filter_hwcf

// CHECK: flow.dispatch.workgroups
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.generic

// -----

func @dont_fuse_conv2d_with_non_identity_map(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>, %offset: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  %3 = linalg.generic {
         indexing_maps = [
           affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
           affine_map<(d0, d1, d2, d3) -> (d3)>,
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
         ins(%2, %offset: tensor<1x112x112x32xf32>, tensor<32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>) {
         ^bb0(%a: f32, %b: f32, %c: f32):
            %sub = subf %a, %b : f32
            linalg.yield %sub : f32
         } -> tensor<1x112x112x32xf32>
  return %3 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func @dont_fuse_conv2d_with_non_identity_map

// CHECK: flow.dispatch.workgroups
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.conv_2d_input_nhwc_filter_hwcf

// CHECK: flow.dispatch.workgroups
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.generic
