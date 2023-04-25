// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions{fuse-multi-use=true}, iree-flow-clone-producers-into-dispatch-regions, iree-flow-form-dispatch-workgroups), cse, canonicalize, cse)" %s | FileCheck %s

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
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[ARG0_DIM0:.+]] = tensor.dim %[[ARG0]], %c0
//  CHECK-DAG:   %[[ARG0_DIM1:.+]] = tensor.dim %[[ARG0]], %c1
//  CHECK-DAG:   %[[ARG1_DIM0:.+]] = tensor.dim %[[ARG1]], %c0
//  CHECK-DAG:   %[[ARG1_DIM1:.+]] = tensor.dim %[[ARG1]], %c1
//  CHECK-DAG:   %[[ARG2_DIM0:.+]] = tensor.dim %[[ARG2]], %c0
//  CHECK-DAG:   %[[ARG2_DIM1:.+]] = tensor.dim %[[ARG2]], %c1
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     [%[[ARG0_DIM0]], %[[ARG0_DIM1]], %[[ARG1_DIM0]], %[[ARG1_DIM1]], %[[ARG2_DIM0]], %[[ARG2_DIM1]]]
// CHECK-SAME:     (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG0_DIM0]], %[[ARG0_DIM1]], %[[ARG1_DIM0]], %[[ARG1_DIM1]], %[[ARG2_DIM0]], %[[ARG2_DIM1]])
// CHECK-NEXT:       %[[ARG3:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK-SAME:       %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK-SAME:       %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>
// CHECK-SAME:       %[[ARG6:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:       %[[ARG7:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:       %[[ARG8:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:       %[[ARG9:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:       %[[ARG10:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:       %[[ARG11:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:     %[[ARG6_W:.+]] = flow.dispatch.workload.ordinal %[[ARG6]] 0
//  CHECK-DAG:     %[[ARG7_W:.+]] = flow.dispatch.workload.ordinal %[[ARG7]] 1
//  CHECK-DAG:     %[[ARG8_W:.+]] = flow.dispatch.workload.ordinal %[[ARG8]] 2
//  CHECK-DAG:     %[[ARG9_W:.+]] = flow.dispatch.workload.ordinal %[[ARG9]] 3
//  CHECK-DAG:     %[[ARG10_W:.+]] = flow.dispatch.workload.ordinal %[[ARG10]] 4
//  CHECK-DAG:     %[[ARG11_W:.+]] = flow.dispatch.workload.ordinal %[[ARG11]] 5
//  CHECK-DAG:     %[[LHS:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG6_W]], %[[ARG7_W]]], strides = [1, 1]
//  CHECK-DAG:     %[[RHS:.+]] = flow.dispatch.tensor.load %[[ARG4]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG8_W]], %[[ARG9_W]]], strides = [1, 1]
//  CHECK-DAG:     %[[OUT:.+]] = flow.dispatch.tensor.load %[[ARG5]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG10_W]], %[[ARG11_W]]], strides = [1, 1]
//      CHECK:     %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:         outs(%[[OUT]] :
//      CHECK:     flow.dispatch.tensor.store %[[GEMM]], %[[ARG5]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG10_W]], %[[ARG11_W]]], strides = [1, 1]
//      CHECK:     count(%[[W0:.+]]: index, %[[W1:.+]]: index, %[[W2:.+]]: index, %[[W3:.+]]: index, %[[W4:.+]]: index, %[[W5:.+]]: index)
//      CHECK:       %[[WX:.+]], %[[WY:.+]], %[[WZ:.+]] = flow.dispatch.workgroup_count_from_slice %[[W0]], %[[W1]], %[[W2]], %[[W3]], %[[W4]], %[[W5]]
//      CHECK:       return %[[WX]], %[[WY]], %[[WZ]]

// -----

func.func @generic_op_alone(%A: tensor<?x?xf32>, %B: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %A, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
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
//      CHECK:   flow.dispatch.workgroups[%[[ARG0_D0]], %[[ARG0_D1]], %[[ARG1_D0]], %[[ARG0_D0]], %[[ARG0_D1]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG1]], %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG1_D0]])
// CHECK-NEXT:       %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?xf32>>,
// CHECK-SAME:       %[[ARG0_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D1_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_D0_CAPTURE:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:       %[[RET0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
//  CHECK-DAG:     %[[ARG0_D0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D0_CAPTURE]] 0
//  CHECK-DAG:     %[[ARG0_D1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D1_CAPTURE]] 1
//  CHECK-DAG:     %[[ARG1_D0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_D0_CAPTURE]] 2
//  CHECK-DAG:     %[[ARG0_D0_W_0:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D0_CAPTURE]] 3
//  CHECK-DAG:     %[[ARG0_D1_W_0:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D1_CAPTURE]] 4
//  CHECK-DAG:     %[[LOAD2:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[ARG0_D0_W]], %[[ARG0_D1_W]]}
//  CHECK-DAG:     %[[LOAD3:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%[[ARG1_D0_W]]}
//  CHECK-DAG:     %[[INIT:.+]] = tensor.empty
//      CHECK:     %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[LOAD2]], %[[LOAD3]] : tensor<?x?xf32>, tensor<?xf32>)
// CHECK-SAME:         outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:     flow.dispatch.tensor.store %[[RESULT]], %[[RET0_CAPTURE]], {{.*}} -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[ARG0_D0_W_0]], %[[ARG0_D1_W_0]]}

// -----

func.func @fuse_matmul_with_fill(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf32>
  %N = tensor.dim %B, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%M, %N) : tensor<?x?xf32>
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
//  CHECK-NEXT:     flow.dispatch.workgroups[%[[ARG0_DIM0]], %[[ARG0_DIM1]], %[[ARG1_DIM0]], %[[ARG1_DIM1]], %[[ARG0_DIM0]], %[[ARG1_DIM1]]]
//  CHECK-SAME:         (%[[ARG0]], %[[ARG1]], %[[ARG0_DIM0]], %[[ARG0_DIM1]], %[[ARG1_DIM0]], %[[ARG1_DIM1]])
//  CHECK-NEXT:         (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>,
//  CHECK-SAME:          %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>,
//  CHECK-SAME:          %[[ARG0_DIM0_CAPTURE:[a-zA-Z0-9_]+]]: index,
//  CHECK-SAME:          %[[ARG0_DIM1_CAPTURE:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:          %[[ARG1_DIM0_CAPTURE:[a-zA-Z0-9_]+]]: index,
//  CHECK-SAME:          %[[ARG1_DIM1_CAPTURE:[a-zA-Z0-9_]+]]: index,
//  CHECK-SAME:          %[[RET0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
//   CHECK-DAG:        %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:        %[[ARG0_DIM0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_DIM0_CAPTURE]] 0
//   CHECK-DAG:        %[[ARG0_DIM1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_DIM1_CAPTURE]] 1
//   CHECK-DAG:        %[[ARG1_DIM0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_DIM0_CAPTURE]] 2
//   CHECK-DAG:        %[[ARG1_DIM1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_DIM1_CAPTURE]] 3
//   CHECK-DAG:        %[[ARG0_DIM0_W_0:.+]] = flow.dispatch.workload.ordinal %[[ARG0_DIM0_CAPTURE]] 4
//   CHECK-DAG:        %[[ARG1_DIM1_W_0:.+]] = flow.dispatch.workload.ordinal %[[ARG1_DIM1_CAPTURE]] 5
//   CHECK-DAG:        %[[LHS:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[ARG0_DIM0_W]], %[[ARG0_DIM1_W]]}
//   CHECK-DAG:        %[[RHS:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[ARG1_DIM0_W]], %[[ARG1_DIM1_W]]}
//   CHECK-DAG:        %[[INIT:.+]] = tensor.empty
//       CHECK:        %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:            ins(%[[ZERO]] :
//  CHECK-SAME:            outs(%[[INIT]] :
//       CHECK:        %[[RESULT:.+]] = linalg.matmul
//  CHECK-SAME:            ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>)
//  CHECK-SAME:            outs(%[[FILL]] : tensor<?x?xf32>)
//       CHECK:        flow.dispatch.tensor.store %[[RESULT]], %[[RET0_CAPTURE]], {{.*}} -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[ARG0_DIM0_W_0]], %[[ARG1_DIM1_W_0]]}
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
  %0 = tensor.empty(%M, %N) : tensor<?x?xf32>
  %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.empty(%M, %K) : tensor<?x?xf32>
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
//       CHECK:     %[[RESULT1:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[ARG0]], %[[M]], %[[K]])
//  CHECK-NEXT:       (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
//  CHECK-SAME:        %[[RET0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
//       CHECK:          %[[ONE:.+]] = arith.constant 1.0
//   CHECK-DAG:          %[[INPUT:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
//   CHECK-DAG:          %[[INIT:.+]] = tensor.empty
//       CHECK:          %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:            ins(%[[INPUT]] : tensor<?x?xf32>)
//  CHECK-SAME:            outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:          flow.dispatch.tensor.store %[[RESULT]], %[[RET0_CAPTURE]]
//       CHECK:          flow.return
//       CHECK:     }
//       CHECK:     flow.dispatch.workgroups
//       CHECK:       %[[ZERO:.+]] = arith.constant 0.0
//       CHECK:       %[[INIT:.+]] = tensor.empty
//       CHECK:       %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:            ins(%[[ZERO]] :
//  CHECK-SAME:            outs(%[[INIT]] :
//       CHECK:       linalg.matmul
//       CHECK:           outs(%[[FILL]] : tensor<?x?xf32>)

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
  %init1 = tensor.empty(%m, %n1) : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul
    ins(%0, %rhs1 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %n2 = tensor.dim %rhs2, %c1 : tensor<4x?xf32>
  %init2 = tensor.empty(%m, %n2) : tensor<?x?xf32>
  %fill2 = linalg.fill ins(%cst : f32) outs(%init2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2= linalg.matmul
    ins(%0, %rhs2 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>
}

//      CHECK: func.func @always_fuse_cast(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<4x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[N2:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[RESULT1:.+]] = flow.dispatch.workgroups[%[[M]], %[[K]], %[[N1]], %[[M]], %[[N1]]]
// CHECK-SAME:     (%[[ARG0]], %[[ARG1]], %[[M]], %[[K]], %[[N1]])
//      CHECK:     tensor.cast
//      CHECK:     flow.return
//      CHECK:   %[[RESULT2:.+]] = flow.dispatch.workgroups[%[[M]], %[[K]], %[[N2]], %[[M]], %[[N2]]]
// CHECK-SAME:     (%[[ARG0]], %[[ARG2]], %[[M]], %[[K]], %[[N2]])
//      CHECK:     tensor.cast
//      CHECK:     flow.return
//      CHECK:   return %[[RESULT1]], %[[RESULT2]]

// -----

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
  %5 = tensor.empty(%3, %4) : tensor<?x?xf32>
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
//  CHECK-DAG: %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG: %[[ARG1_D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG: %[[ARG2_D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG: %[[ARG2_D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//  CHECK-DAG: %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG: %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK: %[[origCC:.+]]:2 = flow.dispatch.workgroups[%[[ARG2_D0]], %[[ARG2_D1]], %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG1_D0]], %[[ARG1_D1]], %[[ARG2_D0]], %[[ARG2_D1]]]
//      CHECK:   %[[ARG2_CAPTURE:.+]]: !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>
// CHECK-SAME:   %[[RESULT_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
//      CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[ARG2_CAPTURE]]
//      CHECK:   %[[STOREVAL:.+]] = linalg.generic
// CHECK-SAME:     outs(%[[LOAD]] : tensor<?x?xf32>)
//      CHECK:   %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:     outs(%[[STOREVAL]] : tensor<?x?xf32>)
//  CHECK-DAG:   flow.dispatch.tensor.store %[[STOREVAL]], %[[RESULT_CAPTURE]]
//  CHECK-DAG:   flow.dispatch.tensor.store %[[GEMM]], %[[ARG2_CAPTURE]]
//      CHECK: return %[[origCC]]#0, %[[origCC]]#1

// -----

func.func @conv2d(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = tensor.empty() : tensor<1x112x112x32xf32>
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
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
//       CHECK:     linalg.conv_2d_nhwc_hwcf
//       CHECK:     flow.return
//       CHECK:   return %[[RESULT]]

// -----

func.func @depthwise_conv2d(%input: tensor<1x113x113x96xf32>, %filter: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : tensor<1x56x56x96xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %4 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%input, %filter : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%2 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  return %4 : tensor<1x56x56x96xf32>
}

// CHECK-LABEL: func.func @depthwise_conv2d
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
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
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[
// CHECK-SAME:        %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]],
// CHECK-SAME:        %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG1_D0]], %[[ARG1_D1]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]],
// CHECK-SAME:        %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG1_D0]], %[[ARG1_D1]])
// CHECK-SAME:       tensor<?x?xf32>{%[[ARG0_D0]], %[[ARG0_D1]]}
// CHECK-SAME:       tensor<?x?xf32>{%[[ARG1_D0]], %[[ARG1_D1]]}
// CHECK-SAME:       -> %[[ARG1]]{%[[ARG1_D0]], %[[ARG1_D1]]}
// CHECK-NEXT:       %[[ARG0_CAPTURE:.+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK-SAME:       %[[ARG1_CAPTURE:.+]]: !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>
// CHECK-SAME:       %[[ARG2_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[ARG3_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[ARG4_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[ARG5_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[ARG0_D0_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[ARG0_D1_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[ARG1_D0_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[ARG1_D1_CAPTURE:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[ARG2_W:.+]] = flow.dispatch.workload.ordinal %[[ARG2_CAPTURE]] 0
//  CHECK-DAG:     %[[ARG3_W:.+]] = flow.dispatch.workload.ordinal %[[ARG3_CAPTURE]] 1
//  CHECK-DAG:     %[[ARG4_W:.+]] = flow.dispatch.workload.ordinal %[[ARG4_CAPTURE]] 2
//  CHECK-DAG:     %[[ARG5_W:.+]] = flow.dispatch.workload.ordinal %[[ARG5_CAPTURE]] 3
//  CHECK-DAG:     %[[ARG0_D0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D0_CAPTURE]] 4
//  CHECK-DAG:     %[[ARG0_D1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D1_CAPTURE]] 5
//  CHECK-DAG:     %[[ARG1_D0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_D0_CAPTURE]] 6
//  CHECK-DAG:     %[[ARG1_D1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_D1_CAPTURE]] 7
//      CHECK:     %[[SRC:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
// CHECK-SAME:         offsets = [0, 0], sizes = [%[[ARG0_D0_W]], %[[ARG0_D1_W]]]
//      CHECK:     flow.dispatch.tensor.store %[[SRC]], %[[ARG1_CAPTURE]]
// CHECK-SAME:         offsets = [%[[ARG2_W]], %[[ARG3_W]]]
// CHECK-SAME:         sizes = [%[[ARG4_W]], %[[ARG5_W]]]
// CHECK-SAME:         !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[ARG1_D0_W]], %[[ARG1_D1_W]]}
//      CHECK:   return %[[RESULT]]

// -----

func.func @fuse_non_tiled_reduction_fill(%input1: tensor<1000xf32>, %input2: tensor<1000xf32>, %offset: tensor<f32>) -> tensor<f32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<f32>
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

//      CHECK: flow.dispatch.workgroups({{.+}}) : (tensor<1000xf32>, tensor<1000xf32>, tensor<f32>) -> tensor<f32> =
// CHECK-NEXT:     (%[[INPUT1:[a-z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<1000xf32>>,
// CHECK-SAME:      %[[INPUT2:[a-z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<1000xf32>>,
// CHECK-SAME:      %[[OFFSET:[a-z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<f32>>,
// CHECK-SAME:      %[[OUTPUT:[a-z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<f32>>) {
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
  %8 = tensor.empty(%arg3) : tensor<1x?xf32>
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
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<i32>>
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG10:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG11:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG12:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<1x?xf32>>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]]
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG5]]
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG6]]
//       CHECK:     %[[INIT:.+]] = tensor.empty
//   CHECK-DAG:     %[[OP1:.+]] = tensor.cast %[[LEAF1]]
//   CHECK-DAG:     %[[OP2:.+]] = tensor.cast %[[LEAF2]]
//   CHECK-DAG:     %[[OP3:.+]] = tensor.extract_slice %[[OP1]][0, 0]
//   CHECK-DAG:     %[[OP4:.+]] = tensor.extract_slice %[[OP1]][0, 10]
//   CHECK-DAG:     %[[OP5:.+]] = tensor.extract_slice %[[OP1]][0, 20]
//       CHECK:     linalg.generic
//  CHECK-SAME:         ins(%[[LEAF3]], %[[OP5]], %[[OP2]], %[[OP4]], %[[OP3]] :
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
  %8 = tensor.empty(%arg3) : tensor<1x?xf32>
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
//  CHECK-NEXT:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<1x?xf32>>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<i32>>
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG10:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG11:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<1x?xf32>>
//       CHECK:     %[[LEAF1:.+]] = flow.dispatch.tensor.load %[[ARG4]], {{.*}}
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG5]], {{.*}}
//       CHECK:     %[[LEAF3:.+]] = flow.dispatch.tensor.load %[[ARG6]], {{.*}}
//       CHECK:     %[[INIT:.+]] = tensor.empty
//   CHECK-DAG:     %[[OP1:.+]] = tensor.cast %[[LEAF1]]
//   CHECK-DAG:     %[[OP3:.+]] = tensor.extract_slice %[[OP1]][0, 0]
//   CHECK-DAG:     %[[OP4:.+]] = tensor.extract_slice %[[OP1]][0, 10]
//   CHECK-DAG:     %[[OP5:.+]] = tensor.extract_slice %[[OP1]][0, 20]
//       CHECK:     linalg.generic
//  CHECK-SAME:         ins(%[[LEAF3]], %[[OP5]], %[[LEAF2]], %[[OP4]], %[[OP3]] :
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
  %255 = tensor.empty() : tensor<9xi1>
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
//  CHECK-SAME:     (%[[ARG2]], %[[UPDATE]])
//  CHECK-NEXT:     (%[[ARG3:.+]]: !flow.dispatch.tensor<readonly:tensor<i32>>,
//  CHECK-SAME:      %[[ARG4:.+]]: !flow.dispatch.tensor<readonly:tensor<18xi32>>,
//  CHECK-SAME:      %[[ARG5:.+]]: !flow.dispatch.tensor<writeonly:tensor<9xi1>>)
//   CHECK-DAG:     %[[C5:.+]] = arith.constant 5 : i32
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[C9:.+]] = arith.constant 9 : i32
//   CHECK-DAG:     %[[ARG4V:.+]] = flow.dispatch.tensor.load %[[ARG3]]
//   CHECK-DAG:     %[[EXTRACT:.+]] = tensor.extract %[[ARG4V]]
//   CHECK-DAG:     %[[CMP1:.+]] = arith.cmpi slt, %[[EXTRACT]]
//   CHECK-DAG:     %[[SELECT1:.+]] = arith.select %[[CMP1]], %[[EXTRACT]], %[[C9]]
//   CHECK-DAG:     %[[CMP2:.+]] = arith.cmpi sgt, %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[SELECT2:.+]] = arith.select %[[CMP2]], %[[SELECT1]], %[[C0]]
//   CHECK-DAG:     %[[INDEX_CAST:.+]] = arith.index_cast %[[SELECT2]]
//   CHECK-DAG:     %[[SLICE:.+]] = flow.dispatch.tensor.load %[[ARG4]], offsets = [%[[INDEX_CAST]]]
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
  %7 = tensor.empty() : tensor<i16>
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
//  CHECK-SAME:     (%[[ARG1]], %[[ARG0]])
//  CHECK-NEXT:     (%[[ARG2:.+]]: !flow.dispatch.tensor<readonly:tensor<i32>>,
//  CHECK-SAME:      %[[ARG3:.+]]: !flow.dispatch.tensor<readonly:tensor<4xi32>>,
//  CHECK-SAME:      %[[ARG4:.+]]: !flow.dispatch.tensor<writeonly:tensor<i16>>
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : i32
//       CHECK:     %[[LEAF2:.+]] = flow.dispatch.tensor.load %[[ARG2]]
//       CHECK:     %[[INIT:.+]] = tensor.empty() : tensor<i16>
//       CHECK:     %[[OP1:.+]] = tensor.extract %[[LEAF2]][] : tensor<i32>
//       CHECK:     %[[OP2:.+]] = arith.cmpi slt, %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP3:.+]] = arith.select %[[OP2]], %[[OP1]], %[[C3]] : i32
//       CHECK:     %[[OP4:.+]] = arith.cmpi sgt, %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP5:.+]] = arith.select %[[OP4]], %[[OP3]], %[[C0]] : i32
//       CHECK:     %[[OP6:.+]] = arith.index_cast %[[OP5]] : i32 to index
//       CHECK:     %[[OP7:.+]] = flow.dispatch.tensor.load %[[ARG3]], offsets = [%[[OP6]]]
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
  %1 = tensor.empty(%0) : tensor<?xi32>
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
//  CHECK-NEXT:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<?xi32>>
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<?xi32>>
//       CHECK:     %[[RESULT:.+]]:2 = linalg.generic
//   CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#0, %[[ARG5]]
//   CHECK-DAG:     flow.dispatch.tensor.store %[[RESULT]]#1, %[[ARG6]]
//       CHECK:   return %[[RESULT_OUT]]#0, %[[RESULT_OUT]]#1

// -----

// TODO: Maybe this test is now not needed anymore.

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
//   CHECK-DAG:   %[[ARG0_D0:.+]] = tensor.dim %[[ARG0]], %c0
//   CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %c1
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[%[[ARG0_D0]], %[[ARG0_D1]], %[[ARG3]]](%[[ARG2]], %[[ARG1]], %[[ARG0]]
//  CHECK-SAME:     ) : (tensor<i32>, tensor<i32>, tensor<?x?xi32>{%[[ARG0_D0]], %[[ARG0_D1]]}
//  CHECK-NEXT:       %[[ARG2_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<i32>>
//  CHECK-SAME:       %[[ARG1_CAPTURE:[a-zA-Z0-9_]*]]: !flow.dispatch.tensor<readonly:tensor<i32>>
//  CHECK-SAME:       %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xi32>>
//  CHECK-SAME:       %[[DEST_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:tensor<1x?xi32>>
//   CHECK-DAG:     flow.dispatch.tensor.load %[[ARG2_CAPTURE]]
//   CHECK-DAG:     flow.dispatch.tensor.load %[[ARG1_CAPTURE]]
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
//       CHECK:     flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
//       CHECK:     flow.dispatch.tensor.store %{{.*}}, %[[DEST_CAPTURE]]
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
  %4 = tensor.empty(%2, %3) : tensor<?x?xf32>
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
      dimension_map = [0]
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
//      CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups[%[[ARG2_D0]], %[[ARG2_D1]], %[[ARG1_D0]], %[[ARG0_D0]], %[[ARG0_D1]]]
// CHECK-SAME:       (%[[ARG2]], %[[ARG1]], %[[ARG0]], %[[ARG2_D0]], %[[ARG2_D1]], %[[ARG1_D0]], %[[ARG0_D0]], %[[ARG0_D1]])
// CHECK-NEXT:       %[[ARG2_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK-SAME:       %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<?x1xi32>>
// CHECK-SAME:       %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>
//  CHECK-DAG:       %[[UPDATE:.+]] = flow.dispatch.tensor.load %[[ARG2_CAPTURE]]
//  CHECK-DAG:       %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]]
//  CHECK-DAG:       %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
//  CHECK-DAG:       %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:               dimension_map = [0]
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
//      CHECK:   %[[RESULT_OUT:.+]]:2 = flow.dispatch.workgroups[
// CHECK-SAME:       %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG0_D2]], %[[ARG1_D0]], %[[ARG1_D1]], %[[ARG1_D2]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG1]], %[[ARG0_D0]], %[[ARG0_D1]], %[[ARG0_D2]], %[[ARG1_D0]], %[[ARG1_D1]], %[[ARG1_D2]])
// CHECK-NEXT:       (%[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:tensor<?x?x?xi32>>
// CHECK-SAME:        %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:tensor<?x?x?xf32>>,
// CHECK-SAME:        %[[ARG0_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D1_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG0_D2_CAPTURE:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:        %[[ARG1_D0_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_D1_CAPTURE:[a-zA-Z0-9_]+]]: index, %[[ARG1_D2_CAPTURE:[a-zA-Z0-9_]+]]: index) {
//  CHECK-DAG:     %[[ARG0_D0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D0_CAPTURE]] 0
//  CHECK-DAG:     %[[ARG0_D1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D1_CAPTURE]] 1
//  CHECK-DAG:     %[[ARG0_D2_W:.+]] = flow.dispatch.workload.ordinal %[[ARG0_D2_CAPTURE]] 2
//  CHECK-DAG:     %[[ARG1_D0_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_D0_CAPTURE]] 3
//  CHECK-DAG:     %[[ARG1_D1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_D1_CAPTURE]] 4
//  CHECK-DAG:     %[[ARG1_D2_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_D2_CAPTURE]] 5
//      CHECK:     %[[OUT1:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
// CHECK-SAME:         offsets = [0, 0, 0], sizes = [%[[ARG0_D0_W]], %[[ARG0_D1_W]], %[[ARG0_D2_W]]]
//      CHECK:     %[[OUT2:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]]
// CHECK-SAME:         offsets = [0, 0, 0], sizes = [%[[ARG1_D0_W]], %[[ARG1_D1_W]], %[[ARG1_D2_W]]]
//      CHECK:     %[[RESULT:.+]]:2 = iree_linalg_ext.sort dimension(0)
// CHECK-SAME:         outs(%[[OUT1]], %[[OUT2]] : tensor<?x?x?xi32>, tensor<?x?x?xf32>)
//      CHECK:     flow.dispatch.tensor.store %[[RESULT]]#0
// CHECK-SAME:         offsets = [0, 0, 0], sizes = [%[[ARG0_D0_W]], %[[ARG0_D1_W]], %[[ARG0_D2_W]]]
//      CHECK:     flow.dispatch.tensor.store %[[RESULT]]#1
// CHECK-SAME:           offsets = [0, 0, 0], sizes = [%[[ARG1_D0_W]], %[[ARG1_D1_W]], %[[ARG1_D2_W]]]
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
      dimension_map = [0]
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
// CHECK-NEXT:     %[[ARG3:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<4xi32>>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:tensor<4x1xi32>>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readwrite:tensor<8xi32>>
//      CHECK:     %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
//      CHECK:     flow.dispatch.tensor.store %[[SCATTER_TILE]], %[[ARG5]], offsets = [0], sizes = [8], strides = [1]
//      CHECK:  return %[[RESULT]]

// -----

// Check that we are distributing along the last three dimensions for NHWC-output pooling op.

func.func @pooling_nwhc_sum_static(%input: tensor<1x33x33x160xf32>) -> tensor<1x3x3x160xf32> {
  %cst = arith.constant 0.0 : f32
  %1 = tensor.empty() : tensor<1x3x3x160xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x3x3x160xf32>) -> tensor<1x3x3x160xf32>
  %3 = tensor.empty() : tensor<11x11xf32>
  %4 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<11> : vector<2xi64>} ins(%input, %3 : tensor<1x33x33x160xf32>, tensor<11x11xf32>) outs(%2 : tensor<1x3x3x160xf32>) -> tensor<1x3x3x160xf32>
  return %4 : tensor<1x3x3x160xf32>
}

// CHECK-LABEL: func.func @pooling_nwhc_sum_static
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups(
//   CHECK-DAG:     %[[INPUT:.+]] = flow.dispatch.tensor.load
//   CHECK-DAG:     %[[EMPTY0:.+]] = tensor.empty() : tensor<1x3x3x160xf32>
//   CHECK-DAG:     %[[EMPTY1:.+]] = tensor.empty() : tensor<11x11xf32>
//       CHECK:     %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[EMPTY0]] :
//       CHECK:     %[[POOL:.+]] = linalg.pooling_nhwc_sum
//  CHECK-SAME:         ins(%[[INPUT]], %[[EMPTY1]] :
//  CHECK-SAME:         outs(%[[FILL]] :
//       CHECK:     flow.dispatch.tensor.store %[[POOL]]
//       CHECK:     flow.return
//       CHECK:   return %[[DISPATCH]]

// -----

func.func @named_op_outs_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst1 = arith.constant -1.0 : f64
  %cstm1 = arith.constant 1.0 : f64
  %c12345 = arith.constant 12345 : i32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
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
//       CHECK:   flow.dispatch.workgroups[%[[D0]], %[[D0]], %[[D1]], %[[D2]]]
//  CHECK-SAME:       tensor<?xi32>{%[[D0]]}
//  CHECK-SAME:       tensor<?x?xi32>{%[[D1]], %[[D2]]}
//  CHECK-NEXT:     !flow.dispatch.tensor<readonly:tensor<?xi32>>
//  CHECK-SAME:     !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>

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
// CHECK-SAME:       [%[[ARG1]], %[[ARG2]], %[[ARG5]], %[[ARG6]], %[[D0]], %[[D1]], %[[ARG3]], %[[ARG4]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG5]], %[[ARG6]], %[[D0]], %[[D1]], %[[ARG3]], %[[ARG4]])
// CHECK-NEXT:     %[[INPUT:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK-SAME:     %[[ARG1_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG5_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG6_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[D0_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[D1_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG4_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
//  CHECK-DAG:     %[[ARG1_W:.+]] = flow.dispatch.workload.ordinal %[[ARG1_CAPTURE]] 0
//  CHECK-DAG:     %[[ARG2_W:.+]] = flow.dispatch.workload.ordinal %[[ARG2_CAPTURE]] 1
//  CHECK-DAG:     %[[ARG5_W:.+]] = flow.dispatch.workload.ordinal %[[ARG5_CAPTURE]] 2
//  CHECK-DAG:     %[[ARG6_W:.+]] = flow.dispatch.workload.ordinal %[[ARG6_CAPTURE]] 3
//  CHECK-DAG:     %[[ARG3_W:.+]] = flow.dispatch.workload.ordinal %[[ARG3_CAPTURE]] 6
//  CHECK-DAG:     %[[ARG4_W:.+]] = flow.dispatch.workload.ordinal %[[ARG4_CAPTURE]] 7
//      CHECK:     %[[SLICE:.+]] = flow.dispatch.tensor.load %[[INPUT]]
// CHECK-SAME:         offsets = [%[[ARG1_W]], %[[ARG2_W]]], sizes = [%[[ARG3_W]], %[[ARG4_W]]], strides = [%[[ARG5_W]], %[[ARG6_W]]]
//      CHECK:     flow.dispatch.tensor.store %[[SLICE]], %[[OUTPUT]],
// CHECK-SAME:         sizes = [%[[ARG3_W]], %[[ARG4_W]]]

// -----

// TODO(ravishankarm): Enable after upstream pad op tiling issues are addressed.
// func.func @tensor.pad(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index,
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
// CHECK-LABEL: func.func @inline_cst2(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<4x2xi32>)
//       CHECK:   flow.dispatch.workgroups
//  CHECK-SAME:     (%[[ARG0]])
//       CHECK:     %[[CST:.+]] = arith.constant dense<[21, 42]> : tensor<2xi32>

// -----

func.func @gemm_unitN(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x1xf32>,
    %arg2 : tensor<?x1xf32>) -> tensor<?x1xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x1xf32>)
      outs(%arg2 : tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}
// CHECK-LABEL: func.func @gemm_unitN(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x1xf32>,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x1xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[K0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[M0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//       CHECK:   flow.dispatch.workgroups[%[[M]], %[[K]], %[[K0]], %[[M0]]]

// -----

func.func @gemm_unitM_unitN(%arg0 : tensor<1x1xf32>, %arg1 : tensor<1x1xf32>,
    %arg2 : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%arg2 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @gemm_unitM_unitN(
//       CHECK:   flow.dispatch.workgroups(
//       CHECK:     linalg.matmul

// -----

func.func @gemm_unitM(%arg0 : tensor<1x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<1x?xf32>) -> tensor<1x?xf32> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<1x?xf32>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}
// CHECK-LABEL: func.func @gemm_unitM(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xf32>,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//   CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[K0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[N0:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK:   flow.dispatch.workgroups[%[[K]], %[[K0]], %[[N0]], %[[N]]]

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
//  CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[ARG0_D4:.+]] = tensor.dim %[[ARG0]], %[[C4]]
//  CHECK-DAG:   %[[ARG0_D5:.+]] = tensor.dim %[[ARG0]], %[[C5]]
//  CHECK-DAG:   %[[ARG0_D7:.+]] = tensor.dim %[[ARG0]], %[[C7]]
//  CHECK-DAG:   %[[ARG1_D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[ARG1_D4:.+]] = tensor.dim %[[ARG1]], %[[C4]]
//  CHECK-DAG:   %[[ARG1_D5:.+]] = tensor.dim %[[ARG1]], %[[C5]]
//  CHECK-DAG:   %[[ARG1_D7:.+]] = tensor.dim %[[ARG1]], %[[C7]]
//      CHECK:   flow.dispatch.workgroups[%[[ARG0_D1]], %[[ARG0_D4]], %[[ARG0_D5]], %[[ARG0_D7]], %[[ARG1_D1]], %[[ARG1_D4]], %[[ARG1_D5]], %[[ARG1_D7]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG1]], %[[ARG0_D1]], %[[ARG0_D4]], %[[ARG0_D5]], %[[ARG0_D7]]

// -----

func.func @dont_fuse_tensor_insert_dest_producer(%arg0 : tensor<2x2xf32>) -> tensor<3x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<0.0> : tensor<3x3xf32>
  %init = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
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
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
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
  %init1 = tensor.empty(%m, %n1) : tensor<?x?xf32>
  %fill1 = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul
    ins(%0, %rhs1 : tensor<?x4xf32>, tensor<4x?xf32>)
    outs(%fill1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %n2 = tensor.dim %rhs2, %c1 : tensor<4x?xf32>
  %init2 = tensor.empty(%m, %n2) : tensor<?x?xf32>
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

// TODO: Maybe this test is now not needed anymore.

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
// CHECK-SAME:       [%[[SOURCE_OFFSET_Y]], %[[SOURCE_OFFSET_X]], %[[SLICE_SIZE]], %[[SOURCE_STRIDE_Y]], %[[SOURCE_STRIDE_X]],
// CHECK-SAME:        %[[DEST_OFFSET_Y]], %[[DEST_OFFSET_X]], %[[DEST_STRIDE_Y]], %[[DEST_STRIDE_X]],
// CHECK-SAME:        %[[SOURCE_D0]], %[[SOURCE_D1]], %[[DEST_D0]], %[[DEST_D1]]]
// CHECK-SAME:       (%[[SOURCE]], %[[SOURCE_OFFSET_Y]], %[[SOURCE_OFFSET_X]],
// CHECK-SAME:        %[[SLICE_SIZE]], %[[SOURCE_STRIDE_Y]], %[[SOURCE_STRIDE_X]],
// CHECK-SAME:        %[[DEST]], %[[DEST_OFFSET_Y]], %[[DEST_OFFSET_X]],
// CHECK-SAME:        %[[DEST_STRIDE_Y]], %[[DEST_STRIDE_X]],
// CHECK-SAME:        %[[SOURCE_D0]], %[[SOURCE_D1]], %[[DEST_D0]], %[[DEST_D1]])
// CHECK-NEXT:       (%[[SOURCE_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>,
// CHECK-SAME:        %[[SOURCE_OFFSET_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_OFFSET_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SLICE_SIZE_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_STRIDE_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_STRIDE_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>,
// CHECK-SAME:        %[[DEST_OFFSET_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_OFFSET_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_STRIDE_Y_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_STRIDE_X_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_D0_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[SOURCE_D1_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_D0_CAPTURE:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:        %[[DEST_D1_CAPTURE:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:     %[[SOURCE_OFFSET_Y_W:.+]] = flow.dispatch.workload.ordinal %[[SOURCE_OFFSET_Y_CAPTURE]] 0
//  CHECK-DAG:     %[[SOURCE_OFFSET_X_W:.+]] = flow.dispatch.workload.ordinal %[[SOURCE_OFFSET_X_CAPTURE]] 1
//  CHECK-DAG:     %[[SLICE_SIZE_W:.+]] = flow.dispatch.workload.ordinal %[[SLICE_SIZE_CAPTURE]] 2
//  CHECK-DAG:     %[[SOURCE_STRIDE_Y_W:.+]] = flow.dispatch.workload.ordinal %[[SOURCE_STRIDE_Y_CAPTURE]] 3
//  CHECK-DAG:     %[[SOURCE_STRIDE_X_W:.+]] = flow.dispatch.workload.ordinal %[[SOURCE_STRIDE_X_CAPTURE]] 4
//  CHECK-DAG:     %[[DEST_OFFSET_Y_W:.+]] = flow.dispatch.workload.ordinal %[[DEST_OFFSET_Y_CAPTURE]] 5
//  CHECK-DAG:     %[[DEST_OFFSET_X_W:.+]] = flow.dispatch.workload.ordinal %[[DEST_OFFSET_X_CAPTURE]] 6
//  CHECK-DAG:     %[[DEST_STRIDE_Y_W:.+]] = flow.dispatch.workload.ordinal %[[DEST_STRIDE_Y_CAPTURE]] 7
//  CHECK-DAG:     %[[DEST_STRIDE_X_W:.+]] = flow.dispatch.workload.ordinal %[[DEST_STRIDE_X_CAPTURE]] 8
//  CHECK-DAG:     %[[SOURCE_D0_W:.+]] = flow.dispatch.workload.ordinal %[[SOURCE_D0_CAPTURE]] 9
//  CHECK-DAG:     %[[SOURCE_D1_W:.+]] = flow.dispatch.workload.ordinal %[[SOURCE_D1_CAPTURE]] 10
//  CHECK-DAG:     %[[DEST_D0_W:.+]] = flow.dispatch.workload.ordinal %[[DEST_D0_CAPTURE]] 11
//  CHECK-DAG:     %[[DEST_D1_W:.+]] = flow.dispatch.workload.ordinal %[[DEST_D1_CAPTURE]] 12
//      CHECK:     %[[SLICE:.+]] = flow.dispatch.tensor.load %[[SOURCE_CAPTURE]]
// CHECK-SAME:         offsets = [%[[SOURCE_OFFSET_Y_W]], %[[SOURCE_OFFSET_X_W]]]
// CHECK-SAME:         sizes = [1, %[[SLICE_SIZE_W]]]
// CHECK-SAME:         strides = [%[[SOURCE_STRIDE_Y_W]], %[[SOURCE_STRIDE_X_W]]]
//      CHECK:     flow.dispatch.tensor.store %[[SLICE]], %[[DEST_CAPTURE]]
// CHECK-SAME:         offsets = [%[[DEST_OFFSET_Y_W]], %[[DEST_OFFSET_X_W]]]
// CHECK-SAME:         sizes = [%[[SLICE_SIZE_W]], 1]
// CHECK-SAME:         strides = [%[[DEST_STRIDE_Y_W]], %[[DEST_STRIDE_X_W]]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func.func @multi_use_producer_fusion(%arg0 : tensor<?x8xf32>, %arg1 : tensor<8x?xf32>,
    %arg2 : tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x8xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<8x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %matmul = linalg.matmul ins(%arg0, %arg1 : tensor<?x8xf32>, tensor<8x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %generic = linalg.generic {
      indexing_maps = [#map0, #map1, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul, %arg2 : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
  return %matmul, %generic : tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func @multi_use_producer_fusion
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x8xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<8x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//      CHECK:   %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups[%[[D0]], %[[D1]], %[[D2]], %[[D0]], %[[D1]]]
// CHECK-SAME:       (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[D0]], %[[D1]], %[[D2]])
// CHECK-NEXT:       %[[ARG0_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?x8xf32>>
// CHECK-SAME:       %[[ARG1_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<8x?xf32>>
// CHECK-SAME:       %[[ARG2_CAPTURE:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?xf32>>
// CHECK-SAME:       %[[D0_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[D1_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[D2_CAPTURE:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[RESULT0:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
// CHECK-SAME:       %[[RESULT1:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
//  CHECK-DAG:     %[[D0_W:.+]] = flow.dispatch.workload.ordinal %[[D0_CAPTURE]] 0
//  CHECK-DAG:     %[[D1_W:.+]] = flow.dispatch.workload.ordinal %[[D1_CAPTURE]] 1
//  CHECK-DAG:     %[[D2_W:.+]] = flow.dispatch.workload.ordinal %[[D2_CAPTURE]] 2
//  CHECK-DAG:     %[[D0_W_0:.+]] = flow.dispatch.workload.ordinal %[[D0_CAPTURE]] 3
//  CHECK-DAG:     %[[D1_W_0:.+]] = flow.dispatch.workload.ordinal %[[D1_CAPTURE]] 4
//  CHECK-DAG:     %[[LHS:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]]
//  CHECK-DAG:     %[[RHS:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]]
//  CHECK-DAG:     %[[BIAS:.+]] = flow.dispatch.tensor.load %[[ARG2_CAPTURE]]
//  CHECK-DAG:     %[[INIT:.+]] = tensor.empty(%[[D0_W_0]], %[[D1_W_0]])
//      CHECK:     %[[FILL:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[INIT]] :
//      CHECK:     %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:         outs(%[[FILL]] :
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[MATMUL]], %[[BIAS]] :
// CHECK-SAME:         outs(%[[INIT]] :
//  CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC]], %[[RESULT0]]
//  CHECK-DAG:     flow.dispatch.tensor.store %[[MATMUL]], %[[RESULT1]]
//      CHECK:   return %[[DISPATCH]]#1, %[[DISPATCH]]#0

// -----

func.func @fft_cst_output(%arg0 : tensor<3x2190x1x512xf32>) -> (tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<1.000000e+00> : tensor<1xf32>
  %cst_0 = arith.constant dense<-0.000000e+00> : tensor<1xf32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<3x2190x1x512xf32>
  %0:2 = iree_linalg_ext.fft ins(%c1, %cst, %cst_0 : index, tensor<1xf32>, tensor<1xf32>)
      outs(%arg0, %cst_1 : tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>) : tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>
  return %0#0, %0#1 : tensor<3x2190x1x512xf32>, tensor<3x2190x1x512xf32>
}
//      CHECK: func @fft_cst_output
// CHECK-SAME:     %[[ARG0:.+]]: tensor<3x2190x1x512xf32>
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK-SAME:       (%[[ARG0]]) : (tensor<3x2190x1x512xf32>) -> (%[[ARG0]], tensor<3x2190x1x512xf32>)
// CHECK-NEXT:     %[[ARG1:.+]]: !flow.dispatch.tensor<readwrite
// CHECK-SAME:     %[[ARG2:.+]]: !flow.dispatch.tensor<writeonly
//      CHECK:       %[[OUT:.+]] = flow.dispatch.tensor.load %[[ARG1]]
//      CHECK:       %[[FFT:.+]]:2 = iree_linalg_ext.fft
// CHECK-SAME:           outs(%[[OUT]],
//  CHECK-DAG:       flow.dispatch.tensor.store %[[FFT]]#0, %[[ARG1]]
//  CHECK-DAG:       flow.dispatch.tensor.store %[[FFT]]#1, %[[ARG2]]

// -----


func.func @fuse_conv2d_elementwise(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>, %offset: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x112x112x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_nhwc_hwcf
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
         outs(%0 : tensor<1x112x112x32xf32>) {
         ^bb0(%a: f32, %b: f32, %c: f32):
            %sub = arith.subf %a, %b : f32
            linalg.yield %sub : f32
         } -> tensor<1x112x112x32xf32>
  return %3 : tensor<1x112x112x32xf32>
}

// Check that
// * linalg.conv is fused together with linalg.generic;
// * linalg.generic's linalg.fill is pulled into the same group;
// * linalg.conv's linalg.fill is pulled into the same group.

// CHECK-LABEL: func.func @fuse_conv2d_elementwise

//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[INIT:.+]] = tensor.empty
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:     outs(%[[INIT]] :
//      CHECK:   %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     outs(%[[FILL]] :
//      CHECK:   linalg.generic
// CHECK-SAME:     ins(%[[CONV]], %{{.+}} : tensor<1x112x112x32xf32>, tensor<32xf32>)
// CHECK-SAME:     outs(%[[INIT]] : tensor<1x112x112x32xf32>)

// -----

func.func @fuse_conv2d_with_multiple_uses(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>, %offset: tensor<32xf32>)
  -> (tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x112x112x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_nhwc_hwcf
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
            %sub = arith.subf %a, %b : f32
            linalg.yield %sub : f32
         } -> tensor<1x112x112x32xf32>
  return %3, %2 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func.func @fuse_conv2d_with_multiple_uses
//       CHECK:   %[[DISPATCH:.+]]:2 = flow.dispatch.workgroups
//  CHECK-NEXT:       %[[OUT1:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
//  CHECK-SAME:       %[[OUT2:.+]]: !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
//       CHECK:     %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//   CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC]], %[[OUT1]]
//   CHECK-DAG:     flow.dispatch.tensor.store %[[CONV]], %[[OUT2]]
//       CHECK:   return %[[DISPATCH]]#0, %[[DISPATCH]]#1

// -----

func.func @dont_fuse_conv2d_with_non_identity_map(%input: tensor<1x225x225x16xf32>, %filter: tensor<3x3x16x32xf32>, %offset: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x112x112x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_nhwc_hwcf
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
            %sub = arith.subf %a, %b : f32
            linalg.yield %sub : f32
         } -> tensor<1x112x112x32xf32>
  return %3 : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: func.func @dont_fuse_conv2d_with_non_identity_map

// CHECK: flow.dispatch.workgroups
// CHECK:   linalg.conv_2d_nhwc_hwcf

// CHECK: flow.dispatch.workgroups
// CHECK:   linalg.generic

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_broadcast_elementwise_unary(%a: tensor<12x16x16xf32>, %b: tensor<12x16x16xf32>) -> tensor<12x16x16xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = tensor.empty() : tensor<12x16xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<12x16xf32>) -> tensor<12x16xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<12x16x16xf32>) outs(%38 : tensor<12x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %780 = arith.maxf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16xf32>
  %40 = tensor.empty() : tensor<12x16x16xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%b, %39 : tensor<12x16x16xf32>, tensor<12x16xf32>) outs(%40 : tensor<12x16x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16x16xf32>
  return %42 : tensor<12x16x16xf32>
}

// There is only one input to the reduction.
// Check that two generic ops are dispatched together.
// The first generic (reduction) is directly used by the second generic (elementwise).

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_unary
//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[RED:.+]] = linalg.generic
//      CHECK:   linalg.generic
//      CHECK-SAME: %[[RED]]

// -----

#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reduction_broadcast_elementwise_binary1(%a1: tensor<128x384xf32>, %a2: tensor<128xf32>, %b: tensor<128x384xf32>) -> tensor<128x384xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = tensor.empty() : tensor<128xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<128xf32>) -> tensor<128xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%a1, %a2 : tensor<128x384xf32>, tensor<128xf32>) outs(%38 : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %585 = arith.subf %arg3, %arg4 : f32
      %586 = arith.mulf %585, %585 : f32
      %587 = arith.addf %586, %arg5 : f32
      linalg.yield %587 : f32
  } -> tensor<128xf32>
  %40 = tensor.empty() : tensor<128x384xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%b, %39 : tensor<128x384xf32>, tensor<128xf32>) outs(%40 : tensor<128x384xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<128x384xf32>
  return %42 : tensor<128x384xf32>
}

// There are two inputs to the reduction and one of them is broadcasted.
// Check that two generic ops are dispatched together.
// The first generic (reduction) is directly used by the second generic (elementwise).

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_binary1
//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[RED:.+]] = linalg.generic
//      CHECK:   linalg.generic
//      CHECK-SAME: %[[RED]]

// -----

#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>

func.func @reduction_broadcast_elementwise_binary2(%a1: tensor<128x384xf32>, %a2: tensor<384xf32>, %b: tensor<128x384xf32>) -> tensor<128x384xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = tensor.empty() : tensor<128xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<128xf32>) -> tensor<128xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map3, #map1], iterator_types = ["parallel", "reduction"]} ins(%a1, %a2 : tensor<128x384xf32>, tensor<384xf32>) outs(%38 : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %585 = arith.subf %arg3, %arg4 : f32
      %586 = arith.mulf %585, %585 : f32
      %587 = arith.addf %586, %arg5 : f32
      linalg.yield %587 : f32
  } -> tensor<128xf32>
  %40 = tensor.empty() : tensor<128x384xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%b, %39 : tensor<128x384xf32>, tensor<128xf32>) outs(%40 : tensor<128x384xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<128x384xf32>
  return %42 : tensor<128x384xf32>
}

// There are two inputs to the reduction and one of them is broadcasted.
// Check that two generic ops are dispatched together.
// The first generic (reduction) is directly used by the second generic (elementwise).

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_binary2
//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[RED:.+]] = linalg.generic
//      CHECK:   linalg.generic
//      CHECK-SAME: %[[RED]]

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_broadcast_elementwise_dynamic(%a: tensor<12x16x?xf32>, %b: tensor<12x16x?xf32>) -> tensor<12x16x?xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = tensor.empty() : tensor<12x16xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<12x16xf32>) -> tensor<12x16xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<12x16x?xf32>) outs(%38 : tensor<12x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %780 = arith.maxf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16xf32>
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %b, %c2 : tensor<12x16x?xf32>
  %40 = tensor.empty(%dim) : tensor<12x16x?xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%b, %39 : tensor<12x16x?xf32>, tensor<12x16xf32>) outs(%40 : tensor<12x16x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16x?xf32>
  return %42 : tensor<12x16x?xf32>
}

// Dynamic shape case is not supported yet by the Vulkan codegen. See #9802.

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_dynamic
//      CHECK: flow.dispatch.workgroups
//      CHECK: linalg.generic
//      CHECK: linalg.generic
//  CHECK-NOT: flow.dispatch.workgroups

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @softmax(%arg0: tensor<12x128x128xf32>) -> tensor<12x128x128xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %0 = tensor.empty() : tensor<12x128xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<12x128xf32>) -> tensor<12x128xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<12x128x128xf32>) outs(%1 : tensor<12x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %7 : f32
    } -> tensor<12x128xf32>
    %3 = tensor.empty() : tensor<12x128x128xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<12x128xf32>) -> tensor<12x128xf32>
    %5:2 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %2 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%3, %4 : tensor<12x128x128xf32>, tensor<12x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %7 = arith.subf %arg1, %arg2 : f32
      %8 = math.exp %7 : f32
      %9 = arith.addf %8, %arg4 : f32
      linalg.yield %8, %9 : f32, f32
    } -> (tensor<12x128x128xf32>, tensor<12x128xf32>)
    %6 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5#0, %5#1 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%3 : tensor<12x128x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %7 = arith.divf %cst, %arg2 : f32
      %8 = arith.mulf %arg1, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<12x128x128xf32>
    return %6 : tensor<12x128x128xf32>
  }
}
// CHECK-LABEL: func @softmax(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12x128x128xf32>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[ARG0]])
//  CHECK-NEXT:     %[[ARG1:.+]]: !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>>
//       CHECK:     %[[LOAD0:.+]] = flow.dispatch.tensor.load %[[ARG1]]
//       CHECK:     %[[FILL0:.+]] = linalg.fill
//       CHECK:     %[[FILL1:.+]] = linalg.fill
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD0]] : tensor<12x128x128xf32>) outs(%[[FILL0]] : tensor<12x128xf32>)
//       CHECK:     %[[GENERIC1:.+]]:2 = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD0]], %[[GENERIC0]] : tensor<12x128x128xf32>, tensor<12x128xf32>)
//  CHECK-SAME:         outs(%{{.*}}, %[[FILL1]] : tensor<12x128x128xf32>, tensor<12x128xf32>)
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC1]]#0, %[[GENERIC1]]#1 :
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC2]]
//       CHECK:     flow.return
//       CHECK:   return %[[DISPATCH]]

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4, d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func @batchnorm_training(%arg0: tensor<12xf32>, %arg1: tensor<12x12x12x12x12xf32>, %arg2: tensor<12xf32>) -> (tensor<12xf32>, tensor<12xf32>, tensor<12xf32>) {
    %cst = arith.constant 1.420000e+00 : f32
    %cst_0 = arith.constant 1.450000e+00 : f32
    %cst_1 = arith.constant 1.300000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<12xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<12xf32>) -> tensor<12xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<12x12x12x12x12xf32>, tensor<12xf32>) outs(%1 : tensor<12xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %4 = arith.subf %arg3, %arg4 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.addf %arg5, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<12xf32>
    %3:3 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %2 : tensor<12xf32>, tensor<12xf32>) outs(%0, %0, %0 : tensor<12xf32>, tensor<12xf32>, tensor<12xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32):
      %4 = arith.divf %arg4, %cst_0 : f32
      %5 = arith.addf %4, %cst_1 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.subf %arg3, %6 : f32
      %8 = arith.mulf %7, %cst : f32
      %9 = arith.subf %arg3, %8 : f32
      linalg.yield %5, %6, %9 : f32, f32, f32
    } -> (tensor<12xf32>, tensor<12xf32>, tensor<12xf32>)
    return %3#0, %3#1, %3#2 : tensor<12xf32>, tensor<12xf32>, tensor<12xf32>
  }
}
// CHECK-LABEL: func @batchnorm_training(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<12xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<12x12x12x12x12xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<12xf32>
//       CHECK:   %[[DISPATCH:.+]]:3 = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[ARG1]], %[[ARG2]], %[[ARG0]])
//  CHECK-NEXT:     %[[ARG3:.+]]: !flow.dispatch.tensor<readonly:tensor<12x12x12x12x12xf32>>
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<12xf32>>
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<12xf32>>
//   CHECK-DAG:     %[[LOAD0:.+]] = flow.dispatch.tensor.load %[[ARG3]]
//   CHECK-DAG:     %[[LOAD1:.+]] = flow.dispatch.tensor.load %[[ARG4]]
//   CHECK-DAG:     %[[LOAD2:.+]] = flow.dispatch.tensor.load %[[ARG5]]
//       CHECK:     %[[FILL:.+]] = linalg.fill
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD0]], %[[LOAD1]] :
//       CHECK:     %[[GENERIC1:.+]]:3 = linalg.generic
//  CHECK-SAME:         ins(%[[LOAD2]], %[[GENERIC0]] :
//   CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC1]]#0
//   CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC1]]#1
//   CHECK-DAG:     flow.dispatch.tensor.store %[[GENERIC1]]#2
//       CHECK:     flow.return
//       CHECK:   return %[[DISPATCH]]#0, %[[DISPATCH]]#1, %[[DISPATCH]]#2

// -----

func.func @set_encoding_op(%arg0 : tensor<?x?xf32>)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> {
  %0 = iree_linalg_ext.set_encoding %arg0
      : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  return %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
}
//      CHECK: func @set_encoding_op
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups[%[[D0]], %[[D1]], %[[D0]], %[[D1]]](%[[ARG0]], %[[D0]], %[[D1]])
// CHECK-NEXT:     %[[INARG:.+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK-SAME:     %[[INDEXARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[INDEXARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OUTARG:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
//  CHECK-DAG:     %[[W0:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG0]] 0
//  CHECK-DAG:     %[[W1:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG1]] 1
//  CHECK-DAG:     %[[W2:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG0]] 2
//  CHECK-DAG:     %[[W3:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG1]] 3
//      CHECK:     %[[LOAD:.+]] = flow.dispatch.tensor.load %[[INARG]]
// CHECK-SAME:         !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[W0]], %[[W1]]}
//      CHECK:     %[[ENCODING:.+]] = iree_linalg_ext.set_encoding %[[LOAD]]
//      CHECK:     flow.dispatch.tensor.store %[[ENCODING]], %[[OUTARG]]
// CHECK-SAME:         sizes = [%[[W2]], %[[W3]]]
// CHECK-SAME:         !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%[[W2]], %[[W3]]}
//      CHECK:     flow.return
//      CHECK:   count(%[[WL0:[a-zA-Z0-9]+]]: index, %[[WL1:[a-zA-Z0-9]+]]: index, %[[WL2:[a-zA-Z0-9]+]]: index, %[[WL3:[a-zA-Z0-9]+]]: index)
//      CHECK:     %[[X:[a-zA-Z0-9]+]], %[[Y:[a-zA-Z0-9]+]], %[[Z:.+]] = flow.dispatch.workgroup_count_from_slice %[[WL0]], %[[WL1]], %[[WL2]], %[[WL3]]
//      CHECK:     flow.return %[[X]], %[[Y]], %[[Z]]
//      CHECK:   return %[[DISPATCH]]

// -----

func.func @unset_encoding_op(%arg0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>)
    -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.unset_encoding %arg0
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func @unset_encoding_op
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups[%[[D0]], %[[D1]], %[[D0]], %[[D1]]](%[[ARG0]], %[[D0]], %[[D1]])
// CHECK-NEXT:       %[[INARG:.+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
// CHECK-SAME:       %[[INDEXARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[INDEXARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[OUTARG:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
//  CHECK-DAG:     %[[D0_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG0]] 0
//  CHECK-DAG:     %[[D1_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG1]] 1
//  CHECK-DAG:     %[[D0_W_0:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG0]] 2
//  CHECK-DAG:     %[[D1_W_0:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG1]] 3
//      CHECK:     %[[LOAD:.+]] = flow.dispatch.tensor.load %[[INARG]]
// CHECK-SAME:         sizes = [%[[D0_W]], %[[D1_W]]]
// CHECK-SAME:         !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%[[D0_W]], %[[D1_W]]}
//      CHECK:     %[[ENCODING:.+]] = iree_linalg_ext.unset_encoding %[[LOAD]]
//      CHECK:     flow.dispatch.tensor.store %[[ENCODING]], %[[OUTARG]]
// CHECK-SAME:         !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[D0_W_0]], %[[D1_W_0]]}
//      CHECK:     flow.return
//      CHECK:   count(%[[WL0:[a-zA-Z0-9]+]]: index, %[[WL1:[a-zA-Z0-9]+]]: index, %[[WL2:[a-zA-Z0-9]+]]: index, %[[WL3:[a-zA-Z0-9]+]]: index)
//      CHECK:     %[[X:[a-zA-Z0-9]+]], %[[Y:[a-zA-Z0-9]+]], %[[Z:.+]] = flow.dispatch.workgroup_count_from_slice %[[WL0]], %[[WL1]], %[[WL2]], %[[WL3]]
//      CHECK:     flow.return %[[X]], %[[Y]], %[[Z]]
//      CHECK:   return %[[DISPATCH]]

// -----

#map = affine_map<()[s0] -> (-s0 + (s0 ceildiv 16) * 16)>
func.func @pad_and_set_encoding_op(%arg0 : tensor<?x?xf32>)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %p0 = affine.apply #map()[%d0]
  %p1 = affine.apply #map()[%d1]
  %pad = tensor.pad %arg0 low[0, 0] high[%p0, %p1] {
    ^bb0(%b0: index, %b1: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  %encoding = iree_linalg_ext.set_encoding %pad
      : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  return %encoding : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> ((s0 ceildiv 16) * 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (-s0 + (s0 ceildiv 16) * 16)>
//      CHECK: func.func @pad_and_set_encoding
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[WLIN0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//  CHECK-DAG:   %[[WLIN1:.+]] = affine.apply #[[MAP0]]()[%[[D1]]]
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups[%[[D1]], %[[D0]], %[[D0]], %[[D1]], %[[WLIN0]], %[[WLIN1]]](%[[D1]], %[[D0]], %[[ARG0]], %[[WLIN0]], %[[WLIN1]])
// CHECK-NEXT:       %[[INDEXARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[INDEXARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[INARG:.+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK-SAME:       %[[PADDED_D0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[PADDED_D1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[OUTARG:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
//  CHECK-DAG:     %[[D1_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG1]] 0
//  CHECK-DAG:     %[[D0_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG0]] 1
//  CHECK-DAG:     %[[D0_W_0:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG0]] 2
//  CHECK-DAG:     %[[D1_W_1:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG1]] 3
//  CHECK-DAG:     %[[PADDED_D0_W:.+]] = flow.dispatch.workload.ordinal %[[PADDED_D0]] 4
//  CHECK-DAG:     %[[PADDED_D1_W:.+]] = flow.dispatch.workload.ordinal %[[PADDED_D1]] 5
//      CHECK:     %[[LOAD:.+]] = flow.dispatch.tensor.load %[[INARG]]
// CHECK-SAME:         !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[D0_W_0]], %[[D1_W_1]]}
//      CHECK:     %[[HIGHPAD1:.+]] = affine.apply #[[MAP1]]()[%[[D1_W]]]
//      CHECK:     %[[HIGHPAD0:.+]] = affine.apply #[[MAP1]]()[%[[D0_W]]]
//      CHECK:     %[[PADDED:.+]] = tensor.pad %[[LOAD]] low[0, 0] high[%[[HIGHPAD0]], %[[HIGHPAD1]]]
//      CHECK:     %[[SET_ENCODING:.+]] = iree_linalg_ext.set_encoding %[[PADDED]]
//      CHECK:     flow.dispatch.tensor.store %[[SET_ENCODING]], %[[OUTARG]]
// CHECK-SAME:         sizes = [%[[PADDED_D0_W]], %[[PADDED_D1_W]]]
// CHECK-SAME:         !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%[[PADDED_D0_W]], %[[PADDED_D1_W]]}
//      CHECK:     flow.return
//      CHECK:   count(%[[WL0:[a-zA-Z0-9]+]]: index, %[[WL1:[a-zA-Z0-9]+]]: index, %[[WL2:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:       %[[WL3:[a-zA-Z0-9]+]]: index, %[[WL4:[a-zA-Z0-9]+]]: index, %[[WL5:[a-zA-Z0-9]+]]: index)
//      CHECK:     %[[X:[a-zA-Z0-9]+]], %[[Y:[a-zA-Z0-9]+]], %[[Z:.+]] = flow.dispatch.workgroup_count_from_slice %[[WL0]], %[[WL1]], %[[WL2]], %[[WL3]], %[[WL4]], %[[WL5]]
//      CHECK:     flow.return %[[X]], %[[Y]], %[[Z]]
//      CHECK:   return %[[DISPATCH]]

// -----

func.func @unset_encoding_and_slice(
    %arg0: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
    %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.unset_encoding %arg0
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0] [%arg1, %arg2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @unset_encoding_and_slice
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups[%[[D0]], %[[D1]], %[[ARG1]], %[[ARG2]]](%[[ARG0]], %[[D0]], %[[D1]], %[[ARG1]], %[[ARG2]])
// CHECK-NEXT:       %[[INARG:.+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
// CHECK-SAME:       %[[INDEXARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[INDEXARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[INDEXARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[INDEXARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:       %[[OUTARG:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
//  CHECK-DAG:     %[[D0_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG0]] 0
//  CHECK-DAG:     %[[D1_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG1]] 1
//  CHECK-DAG:     %[[ARG0_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG2]] 2
//  CHECK-DAG:     %[[ARG1_W:.+]] = flow.dispatch.workload.ordinal %[[INDEXARG3]] 3
//      CHECK:     %[[LOAD:.+]] = flow.dispatch.tensor.load %[[INARG]]
// CHECK-SAME:         sizes = [%[[D0_W]], %[[D1_W]]]
// CHECK-SAME:         !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%[[D0_W]], %[[D1_W]]}
//      CHECK:     %[[ENCODING:.+]] = iree_linalg_ext.unset_encoding %[[LOAD]]
//      CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[ENCODING]][0, 0] [%[[ARG0_W]], %[[ARG1_W]]]
//      CHECK:     flow.dispatch.tensor.store %[[SLICE]], %[[OUTARG]]
// CHECK-SAME:         sizes = [%[[ARG0_W]], %[[ARG1_W]]]
// CHECK-SAME:         !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[ARG0_W]], %[[ARG1_W]]}
//      CHECK:     flow.return

// -----

func.func @gemm_encoded(
    %arg0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
    %arg1 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>,
    %arg2 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>> {
  %0 = linalg.matmul
      ins(%arg0, %arg1
          : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
            tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
      outs(%arg2 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  return %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
}
//      CHECK: func.func @gemm_encoded
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK-NEXT:     %[[LHS_IN:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
// CHECK-SAME:     %[[RHS_IN:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>
// CHECK-SAME:     %[[INIT_IN:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>
//  CHECK-DAG:     %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_IN]]
//  CHECK-DAG:     %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_IN]]
//  CHECK-DAG:     %[[INIT:.+]] = flow.dispatch.tensor.load %[[INIT_IN]]
//      CHECK:     %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:         outs(%[[INIT]] :
//      CHECK:     flow.dispatch.tensor.store %[[GEMM]], %[[INIT_IN]]

// -----

func.func @gemm_fill_encoded(
    %arg0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
    %arg1 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %0 = linalg.matmul
      ins(%arg0, %arg1
          : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
            tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
      outs(%fill : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  return %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
}
//      CHECK: func.func @gemm_fill_encoded
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK-NEXT:     %[[LHS_IN:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
// CHECK-SAME:     %[[RHS_IN:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>
// CHECK-SAME:     %[[RESULT:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>
//  CHECK-DAG:     %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_IN]]
//  CHECK-DAG:     %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_IN]]
//      CHECK:     %[[EMPTY:.+]] = tensor.empty
//      CHECK:     %[[FILL:.+]] = linalg.fill
// CHECK-SAME:         outs(%[[EMPTY]] :
//      CHECK:     %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:         ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:         outs(%[[FILL]] :
//      CHECK:     flow.dispatch.tensor.store %[[GEMM]], %[[RESULT]]

// -----

func.func @extract_slice1(%arg0 : tensor<5x24x48xf32>) -> tensor<4xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 4] [1, 1, 4] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @extract_slice1(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x24x48xf32>)
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C4]] for %[[C1]], %[[C1]], %[[C4]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @clone_fill_ops(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<128x256xf32>, %arg3 : tensor<256x512xf32>)
    -> (tensor<128x512xf32>, tensor<128x512xf32>) {
  %0 = tensor.empty() : tensor<128x512xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %3 = linalg.matmul ins(%arg2, %arg3 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %2, %3 : tensor<128x512xf32>, tensor<128x512xf32>
}
// CHECK-LABEL: func @clone_fill_ops(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<128x256xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<256x512xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9].+]]: tensor<128x256xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9].+]]: tensor<256x512xf32>
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[ARG0]], %[[ARG1]])
//       CHECK:     tensor.empty()
//       CHECK:     linalg.fill
//       CHECK:     linalg.matmul
//       CHECK:   %[[DISPATCH2:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[ARG2]], %[[ARG3]])
//       CHECK:     tensor.empty()
//       CHECK:     linalg.fill
//       CHECK:     linalg.matmul

// -----

func.func @softmax(%source : tensor<12x128x128xf32>) -> tensor<12x128x128xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %1 = tensor.empty() : tensor<12x128xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%source : tensor<12x128x128xf32>) outs(%2 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.maxf %in, %out : f32
    linalg.yield %9 : f32
  } -> tensor<12x128xf32>
  %4 = tensor.empty() : tensor<12x128x128xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%source, %3 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %9 = arith.subf %in, %in_4 : f32
    %10 = math.exp %9 : f32
    linalg.yield %10 : f32
  } -> tensor<12x128x128xf32>
  %6 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<12x128x128xf32>) outs(%6 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.addf %in, %out : f32
    linalg.yield %9 : f32
  } -> tensor<12x128xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %7 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %9 = arith.divf %cst, %in_4 : f32
    %10 = arith.mulf %in, %9 : f32
    linalg.yield %10 : f32
  } -> tensor<12x128x128xf32>
  return %8 : tensor<12x128x128xf32>
}
// CHECK-LABEL: func @softmax(
//  CHECK-SAME:     %[[INPUT:.+]]: tensor<12x128x128xf32>)
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
//  CHECK-SAME:       (%[[INPUT]])
//  CHECK-NEXT:     (%[[ARG0:.+]]: !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>>,
//  CHECK-SAME:      %[[ARG1:.+]]: !flow.dispatch.tensor<writeonly:tensor<12x128x128xf32>>)
//   CHECK-DAG:     %[[CST1:.+]] = arith.constant -3.4
//   CHECK-DAG:     %[[CST2:.+]] = arith.constant 0.0
//       CHECK:     %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//   CHECK-DAG:     %[[EMPTY0:.+]] = tensor.empty() : tensor<12x128xf32>
//   CHECK-DAG:     %[[EMPTY1:.+]] = tensor.empty() : tensor<12x128x128xf32>
//       CHECK:     %[[FILL0:.+]] = linalg.fill ins(%[[CST1]] : f32) outs(%[[EMPTY0]] :
//       CHECK:     %[[FILL1:.+]] = linalg.fill ins(%[[CST2]] : f32) outs(%[[EMPTY0]] :
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[SOURCE]] :
//  CHECK-SAME:         outs(%[[FILL0]] :
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[SOURCE]], %[[GENERIC1]] :
//  CHECK-SAME:         outs(%[[EMPTY1]] :
//       CHECK:     %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC2]] :
//  CHECK-SAME:         outs(%[[FILL1]] :
//       CHECK:     %[[GENERIC4:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC2]], %[[GENERIC3]] :
//  CHECK-SAME:         outs(%[[EMPTY1]] :
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC4]], %[[ARG1]]
//       CHECK:   return %[[RESULT]]
