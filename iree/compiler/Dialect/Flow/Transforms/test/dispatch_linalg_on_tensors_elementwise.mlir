// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-dispatch-linalg-on-tensors-pass -canonicalize -cse %s | IreeFileCheck %s

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


// -----

func @tile_parallel_reduction(%arg0: tensor<7x7x1280xf32>) -> tensor<1280xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1280] : tensor<1280xf32>
  %1 = linalg.fill(%cst, %0) : f32, tensor<1280xf32> -> tensor<1280xf32>
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
// CHECK-NEXT:     %[[OUT:.+]] = linalg.fill(%{{.*}}, %[[INIT]]
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[IN]] : tensor<7x7x?xf32>) outs(%[[OUT]] : tensor<?xf32>)
//      CHECK:     flow.dispatch.tensor.store %[[GENERIC]], %[[ARG2]], {{.*}}

//      CHECK: return %[[REDUCE]]
