// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-dispatch-linalg-on-tensors-pass='tile-sizes=1,2' -cse %s | IreeFileCheck %s

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (2, -d0 + 4)>

// CHECK-LABEL: func @tensor
func @tensor() -> tensor<2x4xf32> {
  //  CHECK-DAG: %[[C1wg:.*]] = constant 1 : index
  //  CHECK-DAG: %[[outerA:.*]] = iree.unfoldable_constant {{.*}} : tensor<2x3xf32>
  //  CHECK-DAG: %[[outerB:.*]] = iree.unfoldable_constant {{.*}} : tensor<3x4xf32>
  //  CHECK-DAG: %[[outerC:.*]] = iree.unfoldable_constant {{.*}} : tensor<2x4xf32>
  %A = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %B = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>
  %C = iree.unfoldable_constant dense<1000.0> : tensor<2x4xf32>

  //      CHECK: flow.dispatch.workgroups[%[[C1wg]], %[[C1wg]], %[[C1wg]]] (%[[outerA]], %[[outerB]], %[[outerC]]) :
  // CHECK-SAME:    (tensor<2x3xf32>, tensor<3x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32> =
  // CHECK-SAME:    (%[[A:[0-9a-z]*]] : !flow.dispatch.input<2x3xf32>,
  // CHECK-SAME:     %[[B:[0-9a-z]*]] : !flow.dispatch.input<3x4xf32>,
  // CHECK-SAME:     %[[C:[0-9a-z]*]] : !flow.dispatch.input<2x4xf32>,
  // CHECK-SAME:     %[[OUT:[0-9a-z]*]] : !flow.dispatch.output<2x4xf32>) {
  //  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
  //  CHECK-DAG:   %[[C1:.*]] = constant 1 : index
  //  CHECK-DAG:   %[[C2:.*]] = constant 2 : index
  //  CHECK-DAG:   %[[C3:.*]] = constant 3 : index
  //  CHECK-DAG:   %[[C4:.*]] = constant 4 : index
  //  CHECK-DAG:   %[[bix:.*]] = flow.dispatch.workgroup.id[0] : index
  //  CHECK-DAG:   %[[bdx:.*]] = flow.dispatch.workgroup.count[0] : index
  //  CHECK-DAG:   %[[biy:.*]] = flow.dispatch.workgroup.id[1] : index
  //  CHECK-DAG:   %[[bdy:.*]] = flow.dispatch.workgroup.count[1] : index
  //      CHECK:   scf.for %[[I:.*]] = %[[biy]] to %[[C2]] step %[[bdy]] {
  // CHECK-NEXT:     %[[bix_scaled:.*]] = muli %[[bix]], %[[C2]] : index
  // CHECK-NEXT:     %[[bdx_scaled:.*]] = muli %[[bdx]], %[[C2]] : index
  // CHECK-NEXT:     scf.for %[[J:.*]] = %[[bix_scaled]] to %[[C4]] step %[[bdx_scaled]] {
  // Canonicalizations not yet powerful enough here.
  // CHECK-NEXT:       %[[MIN_I:.*]] = affine.min{{.*}}(%[[I]])
  // CHECK-NEXT:       %[[AA:.*]] = flow.dispatch.input.load %[[A]],
  // CHECK-SAME:         offsets = [%[[I]], %[[C0]]], sizes = [%[[MIN_I]], %[[C3]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.input<2x3xf32> -> tensor<?x3xf32>
  //
  // Canonicalizations not yet powerful enough here.
  // CHECK-NEXT:       %[[MIN_J:.*]] = affine.min{{.*}}(%[[J]])
  // CHECK-NEXT:       %[[BB:.*]] = flow.dispatch.input.load %arg1,
  // CHECK-SAME:         offsets = [%[[C0]], %[[J]]], sizes = [%[[C3]], %[[MIN_J]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.input<3x4xf32> -> tensor<3x?xf32>
  // CHECK-NEXT:       %[[CC:.*]] = flow.dispatch.input.load %arg2,
  // CHECK-SAME:         offsets = [%[[I]], %[[J]]], sizes = [%[[MIN_I]], %[[MIN_J]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.input<2x4xf32> -> tensor<?x?xf32>
  // CHECK-NEXT:       %[[RES:.*]] = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%[[AA]], %[[BB]] :
  // CHECK-SAME:         tensor<?x3xf32>, tensor<3x?xf32>) outs(%[[CC]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NEXT:       flow.dispatch.output.store %[[RES]], %[[OUT]],
  // CHECK-SAME:         offsets = [%[[I]], %[[J]]], sizes = [%[[MIN_I]], %[[MIN_J]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           tensor<?x?xf32> -> !flow.dispatch.output<2x4xf32>
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT:   flow.return
  %E = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                    outs(%C: tensor<2x4xf32>) -> tensor<2x4xf32>
  return %E : tensor<2x4xf32>
}
