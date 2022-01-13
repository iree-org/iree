// RUN: iree-opt -split-input-file -verify-diagnostics -pass-pipeline="builtin.func(iree-flow-dispatch-linalg-on-tensors-pass, canonicalize, cse)" %s | FileCheck %s

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
// CHECK: #[[MULMAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: func @tile_generic_op_alone
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     [%[[D1]], %[[D0]], %[[C1]]](%[[ARG0]], %[[D0]], %[[D1]], %[[ARG1]], %[[D2]], %[[D0]], %[[D1]])
// CHECK-NEXT:     %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?x?xf32>, %[[ARG0_D0:[a-zA-Z0-9_]+]]: index, %[[ARG0_D1:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:     %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:?xf32>, %[[ARG1_D0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:     %[[RET0_D0:[a-zA-Z0-9_]+]]: index, %[[RET0_D1:[a-zA-Z0-9_]+]]: index, %[[RET_CAPTURE:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:?x?xf32>
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
//  CHECK-DAG:         %[[LOAD2:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?x?xf32>{%[[ARG0_D0]], %[[ARG0_D1]]}
//  CHECK-DAG:         %[[INIT:.+]] = linalg.init_tensor
//  CHECK-DAG:         %[[LOAD3:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], {{.*}} : !flow.dispatch.tensor<readonly:?xf32>{%[[ARG1_D0]]}
//      CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:           ins(%[[LOAD2]], %[[LOAD3]] : tensor<?x?xf32>, tensor<?xf32>)
// CHECK-SAME:           outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:         flow.dispatch.tensor.store %[[RESULT]], %[[RET_CAPTURE]], {{.*}} -> !flow.dispatch.tensor<writeonly:?x?xf32>{%[[RET0_D0]], %[[RET0_D1]]}

// -----


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
//      CHECK:   flow.dispatch.workgroups[%[[D2]], %[[D1]], %[[D0]]]

// -----


// -----

func @tile_parallel_reduction(%arg0: tensor<7x7x1280xf32>) -> tensor<1280xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1280] : tensor<1280xf32>
  %1 = linalg.fill(%cst, %0) : f32, tensor<1280xf32> -> tensor<1280xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg0 : tensor<7x7x1280xf32>) outs(%1 : tensor<1280xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %3 = arith.addf %arg1, %arg2 : f32
    linalg.yield %3 : f32
  } -> tensor<1280xf32>
  return %2 : tensor<1280xf32>
}

//  CHECK-DAG: #[[SIZE_MAP0:.+]] = affine_map<(d0, d1) -> (d0, -d1 + 1280)>
//  CHECK-DAG: #[[SIZE_MAP1:.+]] = affine_map<(d0, d1) -> (-d0 + 1280, d1)>

//      CHECK: func @tile_parallel_reduction
// CHECK-SAME: (%[[INPUT:.+]]: tensor<7x7x1280xf32>)

//  CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG: %[[C1280:.+]] = arith.constant 1280 : index
//      CHECK: %[[REDUCE:.+]] = flow.dispatch.workgroups[%[[C1280]], %[[C1]], %[[C1]]](%[[INPUT]]) : (tensor<7x7x1280xf32>) -> tensor<1280xf32> =
// CHECK-NEXT:     (%[[ARG1:.+]]: !flow.dispatch.tensor<readonly:7x7x1280xf32>, %[[ARG2:.+]]: !flow.dispatch.tensor<writeonly:1280xf32>) {
//      CHECK:   %[[WG_SIZE0:.+]] = flow.dispatch.workgroup.size[0] : index
//      CHECK:   scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
//      CHECK:     %[[SIZE0:.+]] = affine.min #[[SIZE_MAP0]](%[[WG_SIZE0]], %[[IV]])
//      CHECK:     %[[IN:.+]] = flow.dispatch.tensor.load %[[ARG1]],
// CHECK-SAME:       sizes = [7, 7, %[[SIZE0]]]
//      CHECK:     %[[SIZE1:.+]] = affine.min #[[SIZE_MAP1]](%[[IV]], %[[WG_SIZE0]])
//      CHECK:     %[[INIT:.+]] = linalg.init_tensor [%[[SIZE1]]] : tensor<?xf32>
// CHECK-NEXT:     %[[OUT:.+]] = linalg.fill(%{{.*}}, %[[INIT]]
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[IN]] : tensor<7x7x?xf32>) outs(%[[OUT]] : tensor<?xf32>)
//      CHECK:     flow.dispatch.tensor.store %[[GENERIC]], %[[ARG2]], {{.*}}

//      CHECK: return %[[REDUCE]]
