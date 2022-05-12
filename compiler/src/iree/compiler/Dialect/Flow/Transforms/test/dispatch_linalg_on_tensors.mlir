// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="func.func(iree-flow-dispatch-linalg-on-tensors-pass), cse, canonicalize, cse" %s | FileCheck %s

func.func @tile_matmul_alone(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func.func @tile_matmul_alone
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
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
// CHECK-SAME:     %[[ARG10:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG11:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:     %[[LHS:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG4]], %[[ARG5]]], strides = [1, 1]
//  CHECK-DAG:     %[[RHS:.+]] = flow.dispatch.tensor.load %[[ARG6]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG7]], %[[ARG8]]], strides = [1, 1]
//  CHECK-DAG:     %[[OUT:.+]] = flow.dispatch.tensor.load %[[ARG9]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG10]], %[[ARG11]]], strides = [1, 1]
//      CHECK:     %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:         outs(%[[OUT]] :
//      CHECK:     flow.dispatch.tensor.store %[[GEMM]], %[[ARG9]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG4]], %[[ARG8]]], strides = [1, 1]

// -----

func.func @generic_op_alone(%A: tensor<?x?xf32>, %B: tensor<?xf32>) -> tensor<?x?xf32> {
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
//      CHECK: func.func @generic_op_alone(
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

func.func @fuse_matmul_with_fill(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf32>
  %N = tensor.dim %B, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%M, %N] : tensor<?x?xf32>
  %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//       CHECK:   func.func @fuse_matmul_with_fill
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[ARG0_DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:     %[[ARG0_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:     %[[ARG1_DIM0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:     %[[ARG1_DIM1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-NEXT:     flow.dispatch.workgroups[%[[ARG1_DIM1]], %[[ARG0_DIM0]], %[[C1]]]
//  CHECK-SAME:       (%[[ARG0_DIM0]], %[[ARG1_DIM1]], %[[ARG0]], %[[ARG0_DIM1]], %[[ARG1]], %[[ARG1_DIM0]])
//  CHECK-NEXT:       (%[[ARG0_DIM0_CAPTURE:[a-zA-Z0-9_]+]]: index,
//  CHECK-SAME:        %[[ARG1_DIM1_CAPTURE:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:        %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>,
//  CHECK-SAME:        %[[ARG0_DIM1_CAPTURE:[a-zA-Z0-9_]+]]: index,
//  CHECK-SAME:        %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>,
//  CHECK-SAME:        %[[ARG1_DIM0_CAPTURE:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:        %[[RET0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>) {
//       CHECK:        %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:        %[[LHS:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?x?xf32>{%[[ARG0_DIM0_CAPTURE]], %[[ARG0_DIM1_CAPTURE]]}
//   CHECK-DAG:        %[[RHS:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?x?xf32>{%[[ARG1_DIM0_CAPTURE]], %[[ARG1_DIM1_CAPTURE]]}
//   CHECK-DAG:        %[[INIT:.+]] = linalg.init_tensor
//       CHECK:        %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:            ins(%[[ZERO]] :
//  CHECK-SAME:            outs(%[[INIT]] :
//       CHECK:        %[[RESULT:.+]] = linalg.matmul
//  CHECK-SAME:            ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:            outs(%[[FILL]] : tensor<?x?xf32>)
//       CHECK:        flow.dispatch.tensor.store %[[RESULT]], %[[RET0_CAPTURE]], {{.*}} -> !flow.dispatch.tensor<writeonly:?x?xf32>{%[[ARG0_DIM0_CAPTURE]], %[[ARG1_DIM1_CAPTURE]]}
//       CHECK:        flow.return

// -----

func.func @keep_separate_dispatches_for_producer(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf32>
  %N = tensor.dim %B, %c1 : tensor<?x?xf32>
  %K = tensor.dim %A, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%M, %N] : tensor<?x?xf32>
  %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
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
//      CHECK: func.func @keep_separate_dispatches_for_producer
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
//       CHECK:       %[[INIT:.+]] = linalg.init_tensor
//       CHECK:       %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:            ins(%[[ZERO]] :
//  CHECK-SAME:            outs(%[[INIT]] :
//       CHECK:       linalg.matmul
//       CHECK:           outs(%[[FILL]] : tensor<?x?xf32>)

// -----

func.func @tile_4d_generic_op_alone
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
//      CHECK: func.func @tile_4d_generic_op_alone
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[D3:.+]] = tensor.dim %[[ARG0]], %[[C3]]
//  CHECK-DAG:   flow.dispatch.workgroups[%[[D3]], %[[D2]], %[[D1]]]

// -----

func.func @always_fuse_cast
  (%lhs : tensor<?x?xf32>, %rhs1 : tensor<4x?xf32>, %rhs2 : tensor<4x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.cast %lhs : tensor<?x?xf32> to tensor<?x4xf32>
  %m = tensor.dim %0, %c0 : tensor<?x4xf32>
  %n1 = tensor.dim %rhs1, %c1 : tensor<4x?xf32>
  %init1 = linalg.init_tensor [%m, %n1] : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul
    ins(%0, %rhs1 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %n2 = tensor.dim %rhs2, %c1 : tensor<4x?xf32>
  %init2 = linalg.init_tensor [%m, %n2] : tensor<?x?xf32>
  %fill2 = linalg.fill ins(%cst : f32) outs(%init2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2= linalg.matmul
    ins(%0, %rhs2 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

//      CHECK: func.func @always_fuse_cast(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[RHS1:[a-zA-Z0-9_]+]]: tensor<4x?xf32>
// CHECK-SAME:   %[[RHS2:[a-zA-Z0-9_]+]]: tensor<4x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[RESULT1:.+]] = flow.dispatch.workgroups[%[[N1]], %[[M]], %[[C1]]]
// CHECK-SAME:     (%[[M]], %[[N1]], %[[ARG0]], %[[K]], %[[RHS1]])
//      CHECK:     tensor.cast
//      CHECK:     flow.return
//      CHECK:   %[[N2:.+]] = tensor.dim %[[RHS2]], %[[C1]]
//      CHECK:   %[[RESULT2:.+]] = flow.dispatch.workgroups[%[[N2]], %[[M]], %[[C1]]]
// CHECK-SAME:     (%[[M]], %[[N2]], %[[ARG0]], %[[K]], %[[RHS2]])
//      CHECK:     tensor.cast
//      CHECK:     flow.return
//      CHECK:   return %[[RESULT1]], %[[RESULT2]]

// -----

// A subsequent pass is expected to convert linalg.fill and flow.tensor.update into DMA ops.
func.func @dont_fuse_tensor_update_with_fill(
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
  %6 = linalg.fill ins(%0 : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = flow.tensor.update %arg0, %6[%arg2, %arg3] : tensor<?x?xf32>{%1, %2} -> %6 as tensor<?x?xf32>{%3, %4}
  return %7 : tensor<?x?xf32>
}

// CHECK: func.func @dont_fuse_tensor_update_with_fill
// CHECK:   %[[SPLAT:.+]] = flow.tensor.splat
// CHECK:   flow.tensor.update %{{.+}}, %[[SPLAT]]

// -----

func.func @pass_constant_through() -> tensor<2x2x3xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  return %cst : tensor<2x2x3xi32>
}
// CHECK-LABEL: func.func @pass_constant_through()
//       CHECK:   %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<2x2x3xi32>
//       CHECK:   return %[[CST]]

// -----

func.func @fuse_matmul_with_generic_op(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> {
  %f12 = arith.constant 12.0 : f32

  %CC = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%C : tensor<?x?xf32>) {
    ^bb0(%c: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  %D = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%CC: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}
// CHECK-LABEL: func.func @fuse_matmul_with_generic_op
// linalg.generic is fused inside the dispatch region and becomes dead.
//   CHECK-NOT: generic
//     CHECK: flow.dispatch.workgroups
//     CHECK:   %[[CC:.*]] = linalg.generic
//     CHECK:   linalg.matmul{{.*}} outs(%[[CC]]

// -----

func.func @keep_original_producer_uses(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %f12 = arith.constant 12.0 : f32

  // linalg.generic is fused inside the dispatch region and becomes a noop but
  // there is still a use.
  %CC = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%C : tensor<?x?xf32>) {
    ^bb0(%c: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  %D = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%CC: tensor<?x?xf32>) -> tensor<?x?xf32>

  return %D, %CC: tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func.func @keep_original_producer_uses
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
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
//      CHECK:   flow.return
//  CHECK-NOT: linalg.generic
//      CHECK: %[[D:.*]] = flow.dispatch.workgroups
//      CHECK:     linalg.matmul
//      CHECK:     flow.return
//      CHECK: return %[[D]], %[[origCC]]

// -----

func.func @conv2d(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_nhwc_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  return %2 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func.func @conv2d
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32
//   CHECK-DAG:   %[[C112:.+]] = arith.constant 112
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[%[[C32]], %[[C112]], %[[C112]]]
//       CHECK:     linalg.conv_2d_nhwc_hwcf
//       CHECK:     flow.return
//       CHECK:   return %[[RESULT]]

// -----

func.func @depthwise_conv2d(%input: tensor<1x113x113x96xf32>, %filter: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %4 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%input, %filter : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%2 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  return %4 : tensor<1x56x56x96xf32>
}

// CHECK-LABEL: func.func @depthwise_conv2d
//   CHECK-DAG:   %[[C56:.+]] = arith.constant 56
//   CHECK-DAG:   %[[C96:.+]] = arith.constant 96
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[%[[C96]], %[[C56]], %[[C56]]]
//       CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
//       CHECK:     flow.return
//       CHECK:   return %[[RESULT]]

// -----

func.func @subtensor_insert(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index) -> tensor<?x?xf32> {
  %0 = tensor.insert_slice %arg0 into
      %arg1[%arg2, %arg3] [%arg4, %arg5] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func.func @subtensor_insert
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[ARG1_D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[%[[ARG0_D1]], %[[ARG0_D0]], %[[C1]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG1]],
// CHECK-SAME:        %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG1_D0]], %[[ARG1_D1]])
// CHECK-SAME:       tensor<?x?xf32>{%[[ARG0_D0]], %[[ARG0_D1]]}
// CHECK-SAME:       tensor<?x?xf32>{%[[ARG1_D0]], %[[ARG1_D1]]}
// CHECK-SAME:       -> %[[ARG1]]{%[[ARG1_D0]], %[[ARG1_D1]]}
// CHECK-NEXT:     %[[ARG0_CAPTURE:.+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG0_D0_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG0_D1_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1_CAPTURE:.+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
// CHECK-SAME:     %[[ARG2_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG4_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG5_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1_D0_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1_D1_CAPTURE:[a-zA-Z0-9]+]]: index
//      CHECK:     %[[SRC:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG0_D0_CAPTURE]], %[[ARG0_D1_CAPTURE]]]
//      CHECK:     flow.dispatch.tensor.store %[[SRC]], %[[ARG1_CAPTURE]]
// CHECK-SAME:         offsets = [%[[ARG2_CAPTURE]], %[[ARG3_CAPTURE]]]
// CHECK-SAME:         sizes = [%[[ARG4_CAPTURE]], %[[ARG5_CAPTURE]]]
// CHECK-SAME:         !flow.dispatch.tensor<readwrite:?x?xf32>{%[[ARG1_D0_CAPTURE]], %[[ARG1_D1_CAPTURE]]}
//      CHECK:   return %[[RESULT]]

// -----

func.func @fuse_non_tiled_reduction_fill(%input1: tensor<1000xf32>, %input2: tensor<1000xf32>, %offset: tensor<f32>) -> tensor<f32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.init_tensor [] : tensor<f32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
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

// CHECK-LABEL: func.func @fuse_non_tiled_reduction_fill

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

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @inline_dag_1(
    %arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<i32>,
    %arg3 : index) -> tensor<1x?xf32> {
  %0 = tensor.cast %arg0 : tensor<?x?xf32> to tensor<1x?xf32>
  %1 = tensor.extract_slice %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = tensor.cast %1 : tensor<1x?xf32> to tensor<?x?xf32>
  %3 = tensor.cast %arg1 : tensor<?x?xf32> to tensor<1x?xf32>
  %4 = tensor.extract_slice %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = tensor.cast %4  : tensor<1x?xf32> to tensor<?x?xf32>
  %6 = tensor.extract_slice %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = tensor.cast %6 : tensor<1x?xf32> to tensor<?x?xf32>
  %8 = linalg.init_tensor [1, %arg3] : tensor<1x?xf32>
  %9 = linalg.generic {
      indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map1],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg2, %2, %3, %5, %7 : tensor<i32>, tensor<?x?xf32>,
          tensor<1x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%8 : tensor<1x?xf32>) {
      ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32):
        %10 = arith.addf %arg5, %arg6 : f32
        %11 = arith.addf %arg7, %arg8 : f32
        %12 = arith.addf %10, %11 : f32
        %13 = arith.sitofp %arg4 : i32 to f32
        %14 = arith.addf %12, %13 : f32
        linalg.yield %14 : f32
      } -> tensor<1x?xf32>
  return %9 : tensor<1x?xf32>
}
// CHECK-LABEL: func.func @inline_dag_1
//   CHECK-NOT:   linalg.
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG10:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG11:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG12:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:1x?xf32>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]]
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG5]]
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG8]]
//       CHECK:     %[[INIT:.+]] = linalg.init_tensor
//       CHECK:     %[[OP1:.+]] = tensor.cast %[[LEAF3]]
//       CHECK:     %[[OP2:.+]] = tensor.cast %[[LEAF2]]
//       CHECK:     %[[OP3:.+]] = tensor.extract_slice %[[OP1]][0, 0]
//       CHECK:     %[[OP4:.+]] = tensor.extract_slice %[[OP1]][0, 10]
//       CHECK:     %[[OP5:.+]] = tensor.extract_slice %[[OP1]][0, 20]
//       CHECK:     linalg.generic
//  CHECK-SAME:         ins(%[[LEAF1]], %[[OP5]], %[[OP2]], %[[OP4]], %[[OP3]] :
//  CHECK-SAME:         outs(%[[INIT]] :

// -----

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @inline_dag_2(
    %arg0: tensor<?x?xf32>, %arg1 : tensor<1x?xf32>, %arg2 : tensor<i32>,
    %arg3 : index) -> tensor<1x?xf32> {
  %0 = tensor.cast %arg0 : tensor<?x?xf32> to tensor<1x?xf32>
  %1 = tensor.extract_slice %0[0, 20] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %2 = tensor.cast %arg1 : tensor<1x?xf32> to tensor<?x?xf32>
  cf.br ^bb1
^bb1:
  %3 = tensor.cast %1 : tensor<1x?xf32> to tensor<?x?xf32>
  %4 = tensor.extract_slice %0[0, 10] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %5 = tensor.cast %4 : tensor<1x?xf32> to tensor<?x?xf32>
  %6 = tensor.extract_slice %0[0, 0] [1, %arg3] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
  %7 = tensor.cast %6 : tensor<1x?xf32> to tensor<?x?xf32>
  %8 = linalg.init_tensor [1, %arg3] : tensor<1x?xf32>
  %9 = linalg.generic {
      indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map1],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg2, %3, %2, %5, %7 : tensor<i32>, tensor<?x?xf32>,
          tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%8 : tensor<1x?xf32>) {
      ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32):
        %10 = arith.addf %arg5, %arg6 : f32
        %11 = arith.addf %arg7, %arg8 : f32
        %12 = arith.addf %10, %11 : f32
        %13 = arith.sitofp %arg4 : i32 to f32
        %14 = arith.addf %12, %13 : f32
        linalg.yield %14 : f32
      } -> tensor<1x?xf32>
  return %9 : tensor<1x?xf32>
}
// CHECK-LABEL: func.func @inline_dag_2
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x?xf32>
//       CHECK:   flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x?xf32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG10:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG11:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:1x?xf32>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG7]], {{.*}}
//       CHECK:     %[[INIT:.+]] = linalg.init_tensor
//       CHECK:     %[[OP1:.+]] = tensor.cast %[[LEAF3]]
//       CHECK:     %[[OP3:.+]] = tensor.extract_slice %[[OP1]][0, 0]
//       CHECK:     %[[OP4:.+]] = tensor.extract_slice %[[OP1]][0, 10]
//       CHECK:     %[[OP5:.+]] = tensor.extract_slice %[[OP1]][0, 20]
//       CHECK:     linalg.generic
//  CHECK-SAME:         ins(%[[LEAF1]], %[[OP5]], %[[LEAF2]], %[[OP4]], %[[OP3]] :
//  CHECK-SAME:         outs(%[[INIT]] :

// -----

func.func @inline_dag_3(%240 : tensor<9xi32>, %244 : tensor<18xi32>, %247 : tensor<i32>) -> tensor<9xi1> {
  %c9 = arith.constant 9 : index
  %c5_i32 = arith.constant 5 : i32
  %c0_i32 = arith.constant 0 : i32
  %c9_i32 = arith.constant 9 : i32
  %245 = flow.tensor.update %240, %244[%c9] : tensor<9xi32> -> %244 as tensor<18xi32>
  %248 = tensor.extract %247[] : tensor<i32>
  %249 = arith.cmpi slt, %248, %c9_i32 : i32
  %250 = arith.select %249, %248, %c9_i32 : i32
  %251 = arith.cmpi sgt, %250, %c0_i32 : i32
  %252 = arith.select %251, %250, %c0_i32 : i32
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
//       CHECK: func.func @inline_dag_3
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<9xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<18xi32>
//  CHECK-SAME:   %[[ARG2:.+]]: tensor<i32>
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.update %[[ARG0]], %[[ARG1]]
//       CHECK:   flow.dispatch.workgroups
//  CHECK-SAME:     (%[[UPDATE]], %[[ARG2]])
//  CHECK-NEXT:     (%[[ARG3:.+]]: !flow.dispatch.tensor<readonly:18xi32>,
//  CHECK-SAME:      %[[ARG4:.+]]: !flow.dispatch.tensor<readonly:i32>,
//  CHECK-SAME:      %[[ARG5:.+]]: !flow.dispatch.tensor<writeonly:9xi1>)
//   CHECK-DAG:     %[[C5:.+]] = arith.constant 5 : i32
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[C9:.+]] = arith.constant 9 : i32
//   CHECK-DAG:     %[[ARG4V:.+]] = flow.dispatch.tensor.load %[[ARG4]]
//   CHECK-DAG:     %[[EXTRACT:.+]] = tensor.extract %[[ARG4V]]
//   CHECK-DAG:     %[[CMP1:.+]] = arith.cmpi slt, %[[EXTRACT]]
//   CHECK-DAG:     %[[SELECT1:.+]] = arith.select %[[CMP1]], %[[EXTRACT]], %[[C9]]
//   CHECK-DAG:     %[[CMP2:.+]] = arith.cmpi sgt, %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[SELECT2:.+]] = arith.select %[[CMP2]], %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[INDEX_CAST:.+]] = arith.index_cast %[[SELECT2]]
//   CHECK-DAG:     %[[SLICE:.+]] = flow.dispatch.tensor.load %[[ARG3]], offsets = [%[[INDEX_CAST]]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[SLICE]] : tensor<9xi32>)
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC]], %[[ARG5]]
//       CHECK:     flow.return

// -----

#map = affine_map<() -> ()>
func.func @inline_dag_4(%arg0: tensor<4xi32>, %arg1: tensor<i32>) -> tensor<i16> {
  %c3_i32 = arith.constant 3 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.extract %arg1[] : tensor<i32>
  %1 = arith.cmpi slt, %0, %c3_i32 : i32
  %2 = arith.select %1, %0, %c3_i32 : i32
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
  %4 = arith.select %3, %2, %c0_i32 : i32
  %5 = arith.index_cast %4 : i32 to index
  %6 = tensor.extract_slice %arg0[%5] [1] [1] : tensor<4xi32> to tensor<i32>
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %7 = linalg.init_tensor [] : tensor<i16>
  %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%6 : tensor<i32>) outs(%7 : tensor<i16>) {
  ^bb0(%arg2: i32, %arg3: i16):  // no predecessors
    %9 = arith.trunci %arg2 : i32 to i16
    linalg.yield %9 : i16
  } -> tensor<i16>
  return %8 : tensor<i16>
}
// CHECK-LABEL: func.func @inline_dag_4
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<4xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<i32>
//       CHECK:   flow.dispatch.workgroups
//  CHECK-SAME:     (%[[ARG0]], %[[ARG1]])
//  CHECK-NEXT:     (%[[ARG2:.+]]: !flow.dispatch.tensor<readonly:4xi32>
//  CHECK-SAME:      %[[ARG3:.+]]: !flow.dispatch.tensor<readonly:i32>
//  CHECK-SAME:      %[[ARG4:.+]]: !flow.dispatch.tensor<writeonly:i16>
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : i32
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG3]]
//       CHECK:     %[[INIT:.+]] = linalg.init_tensor [] : tensor<i16>
//       CHECK:     %[[OP1:.+]] = tensor.extract %[[LEAF2]][] : tensor<i32>
//       CHECK:     %[[OP2:.+]] = arith.cmpi slt, %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP3:.+]] = arith.select %[[OP2]], %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP4:.+]] = arith.cmpi sgt, %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP5:.+]] = arith.select %[[OP4]], %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP6:.+]] = arith.index_cast %[[OP5]] : i32 to index
//       CHECK:     %[[OP7:.+]] = flow.dispatch.tensor.load %[[ARG2]], offsets = [%[[OP6]]]
//       CHECK:     %[[RES:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[OP7]] : tensor<i32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<i16>) {
//       CHECK:     ^bb0(%[[ARG5:.+]]: i32, %{{.+}}: i16):
//       CHECK:       %[[TRUNC:.+]] = arith.trunci %[[ARG5]] : i32 to i16
//       CHECK:       linalg.yield %[[TRUNC]] : i16
//       CHECK:     } -> tensor<i16>
//       CHECK:     flow.dispatch.tensor.store %[[RES]], %[[ARG4]]

// -----

func.func @multi_result(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
  %cmin = arith.constant -2147483648 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = linalg.init_tensor [%0] : tensor<?xi32>
  %2 = linalg.fill ins(%cmin : i32) outs(%1 : tensor<?xi32>) -> tensor<?xi32>
  %3 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<?xi32>) -> tensor<?xi32>
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
    %6 = arith.select %5, %arg2, %arg4 : i32
    %7 = arith.cmpi eq, %arg2, %arg4 : i32
    %8 = arith.cmpi slt, %arg3, %arg5 : i32
    %9 = arith.select %8, %arg3, %arg5 : i32
    %10 = arith.select %5, %arg3, %arg5 : i32
    %11 = arith.select %7, %9, %10 : i32
    linalg.yield %6, %11 : i32, i32
  } -> (tensor<?xi32>, tensor<?xi32>)
  return %4#0, %4#1 : tensor<?xi32>, tensor<?xi32>
}
// CHECK-LABEL: func.func @multi_result
//       CHECK:   %[[RESULT_OUT:.+]]:2 = flow.dispatch.workgroups
//  CHECK-NEXT:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xi32>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?xi32>
//       CHECK:     %[[RESULT:.+]]:2 = linalg.generic
//   CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#0, %[[ARG5]]
//   CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#1, %[[ARG6]]
//       CHECK:   return %[[RESULT_OUT]]#0, %[[RESULT_OUT]]#1

// -----

func.func @dynamic_slice(%arg0: tensor<?x?xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3 : index) -> tensor<1x?xi32> {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.extract %arg1[] : tensor<i32>
  %1 = arith.cmpi slt, %0, %c1_i32 : i32
  %2 = arith.select %1, %0, %c1_i32 : i32
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
  %4 = arith.select %3, %2, %c0_i32 : i32
  %5 = arith.index_cast %4 : i32 to index
  %6 = tensor.extract %arg2[] : tensor<i32>
  %7 = arith.cmpi slt, %6, %c0_i32 : i32
  %8 = arith.select %7, %6, %c0_i32 : i32
  %9 = arith.cmpi sgt, %8, %c0_i32 : i32
  %10 = arith.select %9, %8, %c0_i32 : i32
  %11 = arith.index_cast %10 : i32 to index
  %12 = tensor.extract_slice %arg0[%5, %11] [1, %arg3] [1, 1] : tensor<?x?xi32> to tensor<1x?xi32>
  return %12 : tensor<1x?xi32>
}
// CHECK-LABEL: func.func @dynamic_slice(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<i32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<i32>
//  CHECK-SAME:   %[[ARG3:.+]]: index
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   flow.tensor.load %[[ARG1]]
//   CHECK-DAG:   flow.tensor.load %[[ARG2]]
//   CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %c0
//   CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %c1
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     arith.select
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     arith.select
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     cmpi
//   CHECK-DAG:     arith.select
//   CHECK-DAG:     arith.select
//   CHECK-DAG:     index_cast
//   CHECK-DAG:     index_cast
//       CHECK:     flow.dispatch.tensor.store
//       CHECK:     flow.return
//       CHECK:   return %[[RESULT]]

// -----

func.func @dynamic_dot() -> !hal.buffer_view attributes {iree.abi.stub} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = flow.tensor.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01], [1.200000e+01, 1.100000e+01, 1.000000e+01], [9.000000e+00, 8.000000e+00, 7.000000e+00], [6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<5x3xf32> -> tensor<?x?xf32>
  %1 = flow.tensor.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01, 1.200000e+01, 1.100000e+01], [1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<3x5xf32> -> tensor<?x?xf32>
  %2 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %3 = tensor.dim %1, %c1 : tensor<?x?xf32>
  %4 = linalg.init_tensor [%2, %3] : tensor<?x?xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = linalg.matmul ins(%0, %1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = tensor.dim %6, %c0 : tensor<?x?xf32>
  %8 = tensor.dim %6, %c1 : tensor<?x?xf32>
  %9 = hal.tensor.export %6 : tensor<?x?xf32>{%7, %8} -> !hal.buffer_view
  return %9 : !hal.buffer_view
}
// CHECK-LABEL: func.func @dynamic_dot()
//   CHECK-NOT:    linalg.fill
//   CHECK-NOT:    linalg.matmul
//       CHECK:    flow.dispatch.workgroups
//       CHECK:      linalg.fill
//       CHECK:      linalg.matmul
//       CHECK:      flow.return
//   CHECK-NOT:    linalg.fill
//   CHECK-NOT:    linalg.matmul
//       CHECK:    return

// -----

func.func @scatter(
    %original : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %update : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
      unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %1 = arith.addf %arg0, %arg1 : f32
        iree_linalg_ext.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func.func @scatter(
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
//  CHECK-DAG:       %[[UPDATE:.+]] = flow.dispatch.tensor.load %[[ARG2_CAPTURE]]
//  CHECK-DAG:       %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]]
//  CHECK-DAG:       %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
//  CHECK-DAG:       %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:               unique_indices(true)
// CHECK-SAME:               ins(%[[UPDATE]], %[[INDICES]] : tensor<?x?xf32>, tensor<?x1xi32>)
// CHECK-SAME:               outs(%[[ORIGINAL]] : tensor<?x?xf32>)
//      CHECK:       flow.dispatch.tensor.store %[[SCATTER]], %[[ARG0_CAPTURE]]
//      CHECK:   return %[[RESULT]] : tensor<?x?xf32>

// -----

func.func @sort_3d(%arg0: tensor<?x?x?xi32>, %arg1 : tensor<?x?x?xf32>)
    -> (tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
  %0, %1 = iree_linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %2 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %2 : i1
      } -> tensor<?x?x?xi32>, tensor<?x?x?xf32>
  return %0, %1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>
}
//      CHECK: func.func @sort_3d(
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
//      CHECK:   %[[RESULT_OUT:.+]]:2 = flow.dispatch.workgroups[%[[ARG0_D2]], %[[ARG0_D1]], %[[C1]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG0_D2]], %[[ARG1]], %[[ARG1_D0]], %[[ARG1_D1]], %[[ARG1_D2]])
// CHECK-NEXT:       (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?x?x?xi32>
// CHECK-SAME:        %[[ARG0_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D1_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D2_CAPTURE:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:        %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:?x?x?xf32>,
// CHECK-SAME:        %[[ARG1_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_D1_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_D2_CAPTURE:[a-zA-Z0-9_]+]]: index) {
//  CHECK-DAG:     %[[OUT1:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
// CHECK-SAME:         offsets = [0, 0, 0], sizes = [%[[ARG0_D0_CAPTURE]], %[[ARG0_D1_CAPTURE]], %[[ARG0_D2_CAPTURE]]]
//  CHECK-DAG:     %[[OUT2:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]]
// CHECK-SAME:         offsets = [0, 0, 0], sizes = [%[[ARG1_D0_CAPTURE]], %[[ARG1_D1_CAPTURE]], %[[ARG1_D2_CAPTURE]]]
//      CHECK:     %[[RESULT:.+]]:2 = iree_linalg_ext.sort dimension(0)
// CHECK-SAME:         outs(%[[OUT1]], %[[OUT2]] : tensor<?x?x?xi32>, tensor<?x?x?xf32>)
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#0
// CHECK-SAME:         offsets = [0, 0, 0], sizes = [%[[ARG0_D0_CAPTURE]], %[[ARG0_D1_CAPTURE]], %[[ARG0_D2_CAPTURE]]]
//  CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#1
// CHECK-SAME:           offsets = [0, 0, 0], sizes = [%[[ARG1_D0_CAPTURE]], %[[ARG1_D1_CAPTURE]], %[[ARG1_D2_CAPTURE]]]
//      CHECK:     flow.return
//      CHECK:   }
//      CHECK:   return %[[RESULT_OUT]]#0, %[[RESULT_OUT]]#1

// -----

func.func @scatter_static(%arg0 : tensor<4xi32>, %arg1 : tensor<4x1xi32>, %arg2 : tensor<8xi32>)
    -> tensor<8xi32>{
  %cst = arith.constant dense<[0, 9, 0, 10, 11, 0, 0, 12]> : tensor<8xi32>
  %cst_0 = arith.constant dense<[9, 10, 11, 12]> : tensor<4xi32>
  %cst_1 = arith.constant dense<[[1], [3], [4], [7]]> : tensor<4x1xi32>
  %cst_2 = arith.constant dense<0> : tensor<8xi32>
  %0 = iree_linalg_ext.scatter
      unique_indices(true)
      ins(%arg0, %arg1 : tensor<4xi32>, tensor<4x1xi32>)
      outs(%arg2 : tensor<8xi32>)  {
    ^bb0(%arg3: i32, %arg4: i32):  // no predecessors
      iree_linalg_ext.yield %arg3 : i32
    } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
//      CHECK: func.func @scatter_static
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4xi32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<4x1xi32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<8xi32>
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
// CHECK-NEXT:     %[[ARG3:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:4xi32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:4x1xi32>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:8xi32>
//      CHECK:     %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
//      CHECK:     flow.dispatch.tensor.store %[[SCATTER_TILE]], %[[ARG5]], offsets = [0], sizes = [8], strides = [1]
//      CHECK:  return %[[RESULT]]

// -----

// Check that we are distributing along the last three dimensions for NHWC-output pooling op.

func.func @pooling_nwhc_sum_static(%input: tensor<1x33x33x160xf32>) -> tensor<1x3x3x160xf32> {
  %cst = arith.constant 0.0 : f32
  %1 = linalg.init_tensor [1, 3, 3, 160] : tensor<1x3x3x160xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x3x3x160xf32>) -> tensor<1x3x3x160xf32>
  %3 = linalg.init_tensor [11, 11] : tensor<11x11xf32>
  %4 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<11> : vector<2xi64>} ins(%input, %3 : tensor<1x33x33x160xf32>, tensor<11x11xf32>) outs(%2 : tensor<1x3x3x160xf32>) -> tensor<1x3x3x160xf32>
  return %4 : tensor<1x3x3x160xf32>
}

// CHECK-LABEL: func.func @pooling_nwhc_sum_static
//  CHECK-DAG:    %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:    %[[C160:.+]] = arith.constant 160 : index
//       CHECK:   flow.dispatch.workgroups[%[[C160]], %[[C3]], %[[C3]]]

// -----

func.func @named_op_outs_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
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
// CHECK-LABEL: func.func @named_op_outs_fusion
//       CHECK:   flow.dispatch.workgroups
//       CHECK:     %[[FILL:.+]] = linalg.fill_rng_2d
//       CHECK:     linalg.matmul
//  CHECK-SAME:       outs(%[[FILL]] : tensor<?x?xf32>)

// -----

func.func @dynamic_slice(%arg0 : i32, %arg1 : i32, %arg2 : tensor<?xi32>,
    %arg3 : tensor<?x?xi32>) -> tensor<?x?xi32>{
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %5 = arith.cmpi slt, %arg0, %c2_i32 : i32
  %6 = arith.select %5, %arg0, %c2_i32 : i32
  %7 = arith.cmpi sgt, %6, %c0_i32 : i32
  %8 = arith.select %7, %6, %c0_i32 : i32
  %9 = arith.index_cast %8 : i32 to index
  %11 = arith.cmpi slt, %arg1, %c0_i32 : i32
  %12 = arith.select %11, %arg1, %c0_i32 : i32
  %13 = arith.cmpi sgt, %12, %c0_i32 : i32
  %14 = arith.select %13, %12, %c0_i32 : i32
  %15 = arith.index_cast %14 : i32 to index
  %d0 = tensor.dim %arg2, %c0 : tensor<?xi32>
  %17 = tensor.insert_slice %arg2 into
      %arg3[%9, %15] [1, %d0] [1, 1] : tensor<?xi32> into tensor<?x?xi32>
  return %17 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @dynamic_slice
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
//  CHECK-SAME:       tensor<?xi32>{%[[D0]]}
//  CHECK-SAME:       tensor<?x?xi32>{%[[D1]], %[[D2]]}
//  CHECK-NEXT:     !flow.dispatch.tensor<readonly:?xi32>
//  CHECK-SAME:     !flow.dispatch.tensor<readwrite:?x?xi32>

// -----

func.func @extract_slice(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index,
    %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> tensor<?x?xf32> {
  %0 = tensor.extract_slice %arg0[%arg1, %arg2] [%arg3, %arg4] [%arg5, %arg6] :
      tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func.func @extract_slice
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:       [%[[ARG4]], %[[ARG3]], %[[C1]]]
// CHECK-SAME:       (%[[ARG0]], %[[D0]], %[[D1]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]])
// CHECK-NEXT:     %[[INPUT:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
// CHECK-SAME:     %[[ARG8:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG9:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG10:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG11:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG12:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG13:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG14:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG15:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>
//      CHECK:     %[[SLICE:.+]] = flow.dispatch.tensor.load %[[INPUT]]
// CHECK-SAME:         offsets = [%[[ARG10]], %[[ARG11]]], sizes = [%[[ARG12]], %[[ARG13]]], strides = [%[[ARG14]], %[[ARG15]]]
//      CHECK:     flow.dispatch.tensor.store %[[SLICE]], %[[OUTPUT]],
// CHECK-SAME:         %[[ARG12]], %[[ARG13]]

// -----

// TODO(ravishankarm): Enable after upstream pad op tiling issues are addressed.
// func.func @pad_tensor(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index,
//     %arg3 : index, %arg4 : index, %arg5 : f32) -> tensor<?x?xf32> {
//   %0 = tensor.pad %arg0 low[%arg1, %arg2] high[%arg3, %arg4] {
//     ^bb0(%arg6 : index, %arg7 : index):
//       tensor.yield %arg5 : f32
//   } :  tensor<?x?xf32> to tensor<?x?xf32>
//   return %0 : tensor<?x?xf32>
// }

// -----

func.func @inline_cst(%arg0 : tensor<4x32xi32>) -> tensor<32xi32> {
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
//      CHECK: func.func @inline_cst(%[[ARG0:.+]]: tensor<4x32xi32>)
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     (%[[ARG0]])
//      CHECK:     %[[CST:.+]] = arith.constant dense<0> : tensor<32xi32>

// -----

func.func @inline_cst2(%arg0 : tensor<4x2xi32>) -> tensor<2xi32> {
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
//      CHECK: func.func @inline_cst2(%[[ARG0:.+]]: tensor<4x2xi32>)
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     (%[[ARG0]])
//      CHECK:     %[[CST:.+]] = arith.constant dense<[21, 42]> : tensor<2xi32>

// -----

func.func @gemm_unitN(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x1xf32>,
    %arg2 : tensor<?x1xf32>) -> tensor<?x1xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x1xf32>)
      outs(%arg2 : tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}
//      CHECK: func.func @gemm_unitN(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x1xf32>,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x1xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0:.+]], %[[C0]]
//      CHECK:   flow.dispatch.workgroups[%[[M]], %[[C1]], %[[C1]]]

// -----

func.func @gemm_unitM_unitN(%arg0 : tensor<1x1xf32>, %arg1 : tensor<1x1xf32>,
    %arg2 : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%arg2 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
//     CHECK: func.func @gemm_unitM_unitN(
//     CHECK:   %[[C1:.+]] = arith.constant 1 : index
//     CHECK:   flow.dispatch.workgroups[%[[C1]], %[[C1]], %[[C1]]]
//     CHECK:     linalg.matmul

// -----

func.func @gemm_unitM(%arg0 : tensor<1x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<1x?xf32>) -> tensor<1x?xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<1x?xf32>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}
//     CHECK: func.func @gemm_unitM(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xf32>,
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1:.+]], %[[C1]]
//      CHECK:   flow.dispatch.workgroups[%[[N]], %[[C1]], %[[C1]]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func.func @unit_dim_generic(%arg0 : tensor<1x?x1x1x?x?x1x?xf32>,
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
//      CHECK: func.func @unit_dim_generic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?x1x1x?x?x1x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x?x1x1x?x?x1x?xf32>)
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//  CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D4:.+]] = tensor.dim %[[ARG0]], %[[C4]]
//  CHECK-DAG:   %[[D5:.+]] = tensor.dim %[[ARG0]], %[[C5]]
//  CHECK-DAG:   %[[D7:.+]] = tensor.dim %[[ARG0]], %[[C7]]
//      CHECK:   flow.dispatch.workgroups[%[[D7]], %[[D5]], %[[D4]]]
// CHECK-SAME:       (%[[ARG0]], %[[D1]], %[[D4]], %[[D5]], %[[D7]]

// -----

func.func @no_fuse_quantized(%arg0 : tensor<?x113x113x64xi8>, %arg1 : tensor<3x3x64xi8>,
    %arg2 : i32, %arg3 : i32) -> tensor<?x56x56x64xi8> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x113x113x64xi8>
  %0 = linalg.init_tensor [%d0, 56, 56, 64] : tensor<?x56x56x64xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x56x56x64xi32>) -> tensor<?x56x56x64xi32>
  %2 =  linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x113x113x64xi8>, tensor<3x3x64xi8>, i32, i32)
      outs(%1 : tensor<?x56x56x64xi32>) -> tensor<?x56x56x64xi32>
  %3 = linalg.init_tensor [%d0, 56, 56, 64] : tensor<?x56x56x64xi8>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%2 : tensor<?x56x56x64xi32>) outs(%3 : tensor<?x56x56x64xi8>) {
    ^bb0(%b0: i32, %b1 : i8):
      %5 = arith.trunci %b0 : i32 to i8
      linalg.yield %5 : i8
    } -> tensor<?x56x56x64xi8>
  return %4 : tensor<?x56x56x64xi8>
}
//     CHECK: func.func @no_fuse_quantized
//     CHECK:   flow.dispatch.workgroups
//     CHECK:   linalg.depthwise_conv_2d_nhwc_hwc_q
// CHECK-NOT:   linalg.generic
//     CHECK:   flow.dispatch.workgroups
//     CHECK:   linalg.generic

// -----

func.func @dont_fuse_tensor_insert_dest_producer(%arg0 : tensor<2x2xf32>) -> tensor<3x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<0.0> : tensor<3x3xf32>
  %init = linalg.init_tensor [2, 2] : tensor<2x2xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %1 = arith.addf %b0, %b0 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2xf32>
  %1 = tensor.insert_slice %0 into %cst[0, 0] [2, 2] [1, 1]
      : tensor<2x2xf32> into tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}
//      CHECK: func.func @dont_fuse_tensor_insert_dest_producer
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x2xf32>
//      CHECK:   %[[CST:.+]] = arith.constant {{.+}} : tensor<3x3xf32>
//      CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.workgroups
//      CHECK:       linalg.generic
//      CHECK:       flow.return
//      CHECK:   %[[DISPATCH2:.+]] = flow.dispatch.workgroups
// CHECK-SAME:       (%[[DISPATCH1]], %[[CST]])
//      CHECK:   return %[[DISPATCH2]]

// -----

func.func @fill_op_alone(%arg0 : index, %arg1 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 42.0 : f32
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func.func @fill_op_alone(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[SPLAT:.+]] = flow.tensor.splat %[[CST]] : tensor<?x?xf32>{%arg0, %arg1}
//      CHECK:   return %[[SPLAT]]

// -----

// Reshapes cannot be fused until #8637 is fixed.
func.func @dont_fuse_reshape(%lhs : tensor<?xf32>, %rhs1 : tensor<4x?xf32>, %rhs2 : tensor<4x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.expand_shape %lhs [[0, 1]] : tensor<?xf32> into tensor<?x4xf32>
  %m = tensor.dim %0, %c0 : tensor<?x4xf32>
  %n1 = tensor.dim %rhs1, %c1 : tensor<4x?xf32>
  %init1 = linalg.init_tensor [%m, %n1] : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul
    ins(%0, %rhs1 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %n2 = tensor.dim %rhs2, %c1 : tensor<4x?xf32>
  %init2 = linalg.init_tensor [%m, %n2] : tensor<?x?xf32>
  %fill2 = linalg.fill ins(%cst : f32) outs(%init2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2= linalg.matmul
    ins(%0, %rhs2 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func.func @dont_fuse_reshape(
// CHECK-SAME:     %[[LHS:.+]]: tensor<?xf32>
//  CHECK-DAG:   %[[RESHAPE:.+]] = flow.tensor.reshape %[[LHS]]
//      CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.workgroups
// CHECK-SAME:       %[[RESHAPE]]
//  CHECK-NOT:     tensor.expand_shape
//      CHECK:     linalg.fill
//      CHECK:     linalg.matmul
//      CHECK:     flow.return
//      CHECK:   %[[DISPATCH2:.+]] = flow.dispatch.workgroups
// CHECK-SAME:       %[[RESHAPE]]
//  CHECK-NOT:     tensor.expand_shape
//      CHECK:     linalg.fill
//      CHECK:     linalg.matmul
//      CHECK:     flow.return
//      CHECK:   return %[[DISPATCH1]], %[[DISPATCH2]]

// -----

func.func @concat_pattern(%src1 : tensor<2x40xf32>, %src2 : tensor<3x40xf32>,
    %dest : tensor<5x40xf32>) -> tensor<5x40xf32> {
  %0 = tensor.insert_slice %src1 into %dest[0, 0] [2, 40] [1, 1]
      : tensor<2x40xf32> into tensor<5x40xf32>
  %1 = tensor.insert_slice %src2 into %0[2, 0] [3, 40] [1, 1]
      : tensor<3x40xf32> into tensor<5x40xf32>
  return %1 : tensor<5x40xf32>
}
//      CHECK: func.func @concat_pattern
// CHECK-SAME:     %[[SRC1:.+]]: tensor<2x40xf32>
// CHECK-SAME:     %[[SRC2:.+]]: tensor<3x40xf32>
// CHECK-SAME:     %[[DEST:.+]]: tensor<5x40xf32>
//      CHECK:   %[[UPDATE1:.+]] = flow.tensor.update %[[SRC1]], %[[DEST]]
//      CHECK:   %[[UPDATE2:.+]] = flow.tensor.update %[[SRC2]], %[[UPDATE1]]
//      CHECK:   return %[[UPDATE2]]

// -----

func.func @generic_tensor_insert(%arg0 : tensor<?x?xf32>,
    %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index,
    %arg5 : index, %arg6 : index, %arg7 : index, %arg8 : index,
    %arg9 : index, %arg10 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.extract_slice %arg0[%arg1, %arg2] [1, %arg3] [%arg4, %arg5] : tensor<?x?xf32> to tensor<?xf32>
  %1 = tensor.insert_slice %0 into %arg10[%arg6, %arg7] [%arg3, 1] [%arg8, %arg9] : tensor<?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func.func @generic_tensor_insert(
// CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[SOURCE_OFFSET_Y:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SOURCE_OFFSET_X:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SLICE_SIZE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SOURCE_STRIDE_Y:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SOURCE_STRIDE_X:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[DEST_OFFSET_Y:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[DEST_OFFSET_X:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[DEST_STRIDE_Y:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[DEST_STRIDE_X:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:     %[[DEST:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[DEST_D0:.+]] = tensor.dim %[[DEST]], %[[C0]]
//  CHECK-DAG:   %[[DEST_D1:.+]] = tensor.dim %[[DEST]], %[[C1]]
//  CHECK-DAG:   %[[SOURCE_D0:.+]] = tensor.dim %[[SOURCE]], %[[C0]]
//  CHECK-DAG:   %[[SOURCE_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]]
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK-SAME:       [%[[SLICE_SIZE]], %[[C1]], %[[C1]]]
// CHECK-SAME:       (%[[SOURCE]], %[[SOURCE_D0]], %[[SOURCE_D1]],
// CHECK-SAME:        %[[SOURCE_OFFSET_Y]], %[[SOURCE_OFFSET_X]]
// CHECK-SAME:        %[[SOURCE_STRIDE_Y]], %[[SOURCE_STRIDE_X]], %[[DEST]]
// CHECK-SAME:        %[[DEST_OFFSET_Y]], %[[DEST_OFFSET_X]], %[[SLICE_SIZE]]
// CHECK-SAME:        %[[DEST_STRIDE_Y]], %[[DEST_STRIDE_X]], %[[DEST_D0]], %[[DEST_D1]])
// CHECK-NEXT:       (%[[SOURCE_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:?x?xf32>,
// CHECK-SAME:        %[[SOURCE_D0_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_D1_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_OFFSET_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_OFFSET_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_STRIDE_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_STRIDE_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:?x?xf32>,
// CHECK-SAME:        %[[DEST_OFFSET_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_OFFSET_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SLICE_SIZE_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_STRIDE_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_STRIDE_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_D0_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_D1_CAPTURE:[a-zA-Z0-9]+]]: index)
//      CHECK:     %[[SLICE:.+]] = flow.dispatch.tensor.load %[[SOURCE_CAPTURE]]
// CHECK-SAME:         offsets = [%[[SOURCE_OFFSET_Y_CAPTURE]], %[[SOURCE_OFFSET_X_CAPTURE]]]
// CHECK-SAME:         sizes = [1, %[[SLICE_SIZE_CAPTURE]]]
// CHECK-SAME:         strides = [%[[SOURCE_STRIDE_Y_CAPTURE]], %[[SOURCE_STRIDE_X_CAPTURE]]]
//      CHECK:     flow.dispatch.tensor.store %[[SLICE]], %[[DEST_CAPTURE]]
// CHECK-SAME:         offsets = [%[[DEST_OFFSET_Y_CAPTURE]], %[[DEST_OFFSET_X_CAPTURE]]]
// CHECK-SAME:         sizes = [%[[SLICE_SIZE_CAPTURE]], 1]
// CHECK-SAME:         strides = [%[[DEST_STRIDE_Y_CAPTURE]], %[[DEST_STRIDE_X_CAPTURE]]]
