// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-dispatch-linalg-on-tensors-pass -canonicalize -cse %s | IreeFileCheck %s

func @tensor(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @tensor
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     (%[[ARG0]], %[[ARG1]], %[[ARG2]])
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] : !flow.dispatch.input<?x?xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]] : !flow.dispatch.input<?x?xf32>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]] : !flow.dispatch.input<?x?xf32>
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] : !flow.dispatch.output<?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:     %[[WGSIZE_X:.+]] = flow.dispatch.workgroup.size[0]
//  CHECK-DAG:     %[[WGSIZE_Y:.+]] = flow.dispatch.workgroup.size[1]
//  CHECK-DAG:     %[[WGID_X:.+]] = flow.dispatch.workgroup.id[0]
//  CHECK-DAG:     %[[WGID_Y:.+]] = flow.dispatch.workgroup.id[1]
//  CHECK-DAG:     %[[WGCOUNT_X:.+]] = flow.dispatch.workgroup.count[0]
//  CHECK-DAG:     %[[WGCOUNT_Y:.+]] = flow.dispatch.workgroup.count[1]
//      CHECK:     %[[OFFSET_Y:.+]] = muli %[[WGSIZE_Y]], %[[WGID_Y]]
//      CHECK:     %[[STEP_Y:.+]] = muli %[[WGSIZE_Y]], %[[WGCOUNT_Y]]
//      CHECK:     scf.for %[[ARG7:.+]] = %[[OFFSET_Y]]
// CHECK-SAME:       to %{{.+}} step %[[STEP_Y]]
//      CHECK:       %[[OFFSET_X:.+]] = muli %[[WGSIZE_X]], %[[WGID_X]]
//      CHECK:       %[[STEP_X:.+]] = muli %[[WGSIZE_X]], %[[WGCOUNT_X]]
//      CHECK:       scf.for %[[ARG8:.+]] = %[[OFFSET_X]]
// CHECK-SAME:         to %{{.+}} step %[[STEP_X]]
//      CHECK:         %[[LHS:.+]] = flow.dispatch.input.load %[[ARG3]]
// CHECK-SAME:           offsets = [%[[ARG7]], %[[C0]]]
//      CHECK:         %[[RHS:.+]] = flow.dispatch.input.load %[[ARG4]]
// CHECK-SAME:           offsets = [%[[C0]], %[[ARG8]]]
//      CHECK:         %[[INIT:.+]] = flow.dispatch.input.load %[[ARG5]]
// CHECK-SAME:           offsets = [%[[ARG7]], %[[ARG8]]]
//      CHECK:         %[[RESULT:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:         flow.dispatch.output.store %[[RESULT]], %[[ARG6]]
// CHECK-SAME:           offsets = [%[[ARG7]], %[[ARG8]]]

// -----

func @generic_op(%A: tensor<?x?xf32>, %B: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = dim %A, %c0 : tensor<?x?xf32>
  %d1 = dim %A, %c1 : tensor<?x?xf32>
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
//      CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0 - d1)>
//      CHECK: func @generic_op
//      CHECK:   flow.dispatch.workgroups
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]] : index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] : index
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]] : !flow.dispatch.input<?x?xf32>
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]] : !flow.dispatch.input<?xf32>
// CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] : !flow.dispatch.output<?x?xf32>
//  CHECK-DAG:     %[[WG_SIZE_X:.+]] = flow.dispatch.workgroup.size[0]
//  CHECK-DAG:     %[[WG_SIZE_Y:.+]] = flow.dispatch.workgroup.size[1]
//      CHECK:     scf.for %[[IV0:[a-zA-Z0-9_]+]]
//      CHECK:       scf.for %[[IV1:.[a-zA-Z0-9_]+]]
//      CHECK:       %[[V1:.+]] = flow.dispatch.input.load %[[ARG4]]
//      CHECK:       %[[V2:.+]] = flow.dispatch.input.load %[[ARG5]]
//      CHECK:       %[[D0:.+]] = affine.min #[[MAP1]](%[[ARG2]], %[[IV0]], %[[WG_SIZE_Y]])
//      CHECK:       %[[D1:.+]] = affine.min #[[MAP1]](%[[ARG3]], %[[IV1]], %[[WG_SIZE_X]])
//      CHECK:       %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//      CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[V1]], %[[V2]] : tensor<?x?xf32>, tensor<?xf32>)
// CHECK-SAME:         outs(%[[INIT]] : tensor<?x?xf32>)
//      CHECK:         flow.dispatch.output.store %[[RESULT]], %[[ARG6]], offsets = [%[[IV0]], %[[IV1]]], sizes = [%[[D0]], %[[D1]]]
