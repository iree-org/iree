// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-dispatch-linalg-on-tensors-pass -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,2" -iree-flow-dispatch-linalg-on-tensors-enable-fusion-always -canonicalize -cse %s | IreeFileCheck %s

// CHECK: #[[mul_map:.+]] = affine_map<()[s0] -> (s0 * 2)>

// CHECK: func @tensor
func @tensor() -> tensor<2x4xf32> {
  //  CHECK-DAG: %[[C1wg:.*]] = constant 1 : index
  //  CHECK-DAG: %[[C2wg:.*]] = constant 2 : index
  //  CHECK-DAG: %[[C4wg:.*]] = constant 4 : index
  //  CHECK-DAG: %[[outerA:.*]] = iree.do_not_optimize{{.*}} : tensor<2x3xf32>
  //  CHECK-DAG: %[[outerB:.*]] = iree.do_not_optimize{{.*}} : tensor<3x4xf32>
  //  CHECK-DAG: %[[outerC:.*]] = iree.do_not_optimize{{.*}} : tensor<2x4xf32>
  %A = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %B = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>
  %C = iree.unfoldable_constant dense<1000.0> : tensor<2x4xf32>

  // %[[C2]] will be handled by a later RematerializeDispatchConstants
  //      CHECK: flow.dispatch.workgroups[%[[C4wg]], %[[C2wg]], %[[C1wg]]] (%[[outerA]], %[[outerB]], %[[outerC]]) :
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
  // CHECK-NEXT:     %[[bix_scaled:.*]] = affine.apply #[[mul_map]]()[%[[bix]]]
  // CHECK-NEXT:     %[[bdx_scaled:.*]] = affine.apply #[[mul_map]]()[%[[bdx]]]
  // CHECK-NEXT:     scf.for %[[J:.*]] = %[[bix_scaled]] to %[[C4]] step %[[bdx_scaled]] {
  // Canonicalizations not yet powerful enough here.
  // CHECK-NEXT:       %[[MIN_I:.*]] = affine.min{{.*}}(%[[I]])
  // CHECK-NEXT:       %[[AA:.*]] = flow.dispatch.input.load %[[A]],
  // CHECK-SAME:         offsets = [%[[I]], %[[C0]]], sizes = [%[[MIN_I]], %[[C3]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.input<2x3xf32> -> tensor<?x3xf32>
  //
  // Canonicalizations not yet powerful enough here.
  // CHECK-NEXT:       %[[MIN_J:.*]] = affine.min{{.*}}(%[[J]])
  // CHECK-NEXT:       %[[BB:.*]] = flow.dispatch.input.load %[[B]],
  // CHECK-SAME:         offsets = [%[[C0]], %[[J]]], sizes = [%[[C3]], %[[MIN_J]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.input<3x4xf32> -> tensor<3x?xf32>
  // CHECK-NEXT:       %[[CC:.*]] = flow.dispatch.input.load %[[C]],
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

// CHECK-LABEL: func @tensor2
func @tensor2(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> attributes {iree.module.export}
{
  %f12 = constant 12.0 : f32

  // linalg.generic is fused inside the dispatch region and becomes a noop.
  // CHECK-NOT: generic
  %AA = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%A : tensor<?x?xf32>) {
    ^bb0(%a: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  // CHECK: flow.dispatch.workgroups
  // CHECK:  scf.for
  // CHECK:    scf.for
  // CHECK:      %[[AA:.*]] = linalg.generic
  // CHECK:      linalg.matmul{{.*}} ins(%[[AA]], %{{.}}
  %D = linalg.matmul ins(%AA, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}

// CHECK-LABEL: func @tensor3
func @tensor3(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> attributes {iree.module.export}
{
  %f12 = constant 12.0 : f32

  // linalg.generic is fused inside the dispatch region and becomes a noop.
  // CHECK-NOT: generic
  %BB = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%B : tensor<?x?xf32>) {
    ^bb0(%b: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  // CHECK: flow.dispatch.workgroups
  // CHECK:  scf.for
  // CHECK:    scf.for
  // CHECK:      %[[BB:.*]] = linalg.generic{{.*}}
  // CHECK:      linalg.matmul{{.*}} ins(%{{.*}}, %[[BB]]
  %D = linalg.matmul ins(%A, %BB: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}


// CHECK-LABEL: func @tensor4
func @tensor4(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
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

//       CHECK: func @tensor5
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
func @tensor5(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> (tensor<?x?xf32>, tensor<?x?xf32>) attributes {iree.module.export}
{
  %f12 = constant 12.0 : f32
  //  CHECK-DAG: %[[C0:.+]] = constant 0 : index
  //  CHECK-DAG: %[[C1:.+]] = constant 1 : index
  //  CHECK-DAG: %[[D0:.+]] = dim %[[ARG2]], %[[C0]]
  //  CHECK-DAG: %[[D1:.+]] = dim %[[ARG2]], %[[C1]]
  //      CHECK: %[[origCC:.+]] = flow.dispatch.workgroups[%[[D1]], %[[D0]], %[[C1]]] (%[[ARG2]])
  // CHECK-SAME:   %[[ARG3:.+]] : !flow.dispatch.input<?x?xf32>
  // CHECK-SAME:   %[[ARG4:.+]] : !flow.dispatch.output<?x?xf32>
  //      CHECK:   %[[LOAD:.+]] = flow.dispatch.input.load %[[ARG3]]
  //      CHECK:   %[[STOREVAL:.+]] = linalg.generic
  // CHECK-SAME:     outs(%[[LOAD]] : tensor<?x?xf32>)
  //      CHECK:   flow.dispatch.output.store %[[STOREVAL]], %[[ARG4]]

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
