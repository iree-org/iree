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
  //      CHECK: flow.dispatch.workgroups[%[[C4wg]], %[[C2wg]], %[[C1wg]]](%[[outerA]], %[[outerB]], %[[outerC]]) :
  // CHECK-SAME:    (tensor<2x3xf32>, tensor<3x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32> =
  // CHECK-NEXT:    (%[[A:[0-9a-z]*]]: !flow.dispatch.tensor<readonly:2x3xf32>,
  // CHECK-SAME:     %[[B:[0-9a-z]*]]: !flow.dispatch.tensor<readonly:3x4xf32>,
  // CHECK-SAME:     %[[C:[0-9a-z]*]]: !flow.dispatch.tensor<readonly:2x4xf32>,
  // CHECK-SAME:     %[[OUT:[0-9a-z]*]]: !flow.dispatch.tensor<writeonly:2x4xf32>) {
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
  // CHECK-NEXT:       %[[AA:.*]] = flow.dispatch.tensor.load %[[A]],
  // CHECK-SAME:         offsets = [%[[I]], %[[C0]]], sizes = [%[[MIN_I]], %[[C3]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.tensor<readonly:2x3xf32> -> tensor<?x3xf32>
  //
  // Canonicalizations not yet powerful enough here.
  // CHECK-NEXT:       %[[MIN_J:.*]] = affine.min{{.*}}(%[[J]])
  // CHECK-NEXT:       %[[BB:.*]] = flow.dispatch.tensor.load %[[B]],
  // CHECK-SAME:         offsets = [%[[C0]], %[[J]]], sizes = [%[[C3]], %[[MIN_J]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.tensor<readonly:3x4xf32> -> tensor<3x?xf32>
  // CHECK-NEXT:       %[[CC:.*]] = flow.dispatch.tensor.load %[[C]],
  // CHECK-SAME:         offsets = [%[[I]], %[[J]]], sizes = [%[[MIN_I]], %[[MIN_J]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           !flow.dispatch.tensor<readonly:2x4xf32> -> tensor<?x?xf32>
  // CHECK-NEXT:       %[[RES:.*]] = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%[[AA]], %[[BB]] :
  // CHECK-SAME:         tensor<?x3xf32>, tensor<3x?xf32>) outs(%[[CC]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NEXT:       flow.dispatch.tensor.store %[[RES]], %[[OUT]],
  // CHECK-SAME:         offsets = [%[[I]], %[[J]]], sizes = [%[[MIN_I]], %[[MIN_J]]], strides = [%[[C1]], %[[C1]]] :
  // CHECK-SAME:           tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:2x4xf32>
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT:   flow.return
  %E = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                    outs(%C: tensor<2x4xf32>) -> tensor<2x4xf32>
  return %E : tensor<2x4xf32>
}

// -----

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

// -----

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

// -----

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

// -----

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
  //      CHECK: %[[origCC:.+]] = flow.dispatch.workgroups[%[[D1]], %[[D0]], %[[C1]]](%[[ARG2]])
  // CHECK-NEXT:   %[[ARG3:.+]]: !flow.dispatch.tensor<readwrite:?x?xf32>
  //      CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[ARG3]]
  //      CHECK:   %[[STOREVAL:.+]] = linalg.generic
  // CHECK-SAME:     outs(%[[LOAD]] : tensor<?x?xf32>)
  //      CHECK:   flow.dispatch.tensor.store %[[STOREVAL]], %[[ARG3]]

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
// CHECK-NEXT:     flow.dispatch.tensor.store %[[TENSOR]], %[[OUTPUT]]
// CHECK-NEXT:     flow.return
//
//      CHECK:   %[[PAD:.+]] = flow.dispatch.workgroups[{{.+}}](%[[INPUT]], %[[FILL]]) : (tensor<1x224x224x3xf32>, tensor<1x225x225x3xf32>) -> %[[FILL]] =
// CHECK-NEXT:       (%[[SRC:.+]]: !flow.dispatch.tensor<readonly:1x224x224x3xf32>, %[[DST:.+]]: !flow.dispatch.tensor<readwrite:1x225x225x3xf32>) {
// CHECK-NEXT:     %[[SRC_TENSOR:.+]] = flow.dispatch.tensor.load %[[SRC]] : !flow.dispatch.tensor<readonly:1x224x224x3xf32> -> tensor<1x224x224x3xf32>
// CHECK-NEXT:     %[[DST_TENSOR:.+]] = flow.dispatch.tensor.load %[[DST]] : !flow.dispatch.tensor<readwrite:1x225x225x3xf32> -> tensor<1x225x225x3xf32>
// CHECK-NEXT:     %[[INSERT:.+]] = subtensor_insert %[[SRC_TENSOR]] into %[[DST_TENSOR]][0, 0, 0, 0] [1, 224, 224, 3] [1, 1, 1, 1]
// CHECK-NEXT:     flow.dispatch.tensor.store %[[INSERT]], %[[DST]] : tensor<1x225x225x3xf32> -> !flow.dispatch.tensor<readwrite:1x225x225x3xf32>
// CHECK-NEXT:     flow.return
//
//      CHECK:   return %[[PAD]] : tensor<1x225x225x3xf32>

// -----

func @reduce(%arg0: tensor<1x7x7x1280xf32>) -> tensor<1x1280xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 1280] : tensor<1x1280xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x1280xf32>, f32 -> tensor<1x1280xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2, d0)>, affine_map<(d0, d1, d2) -> (0, d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg0 : tensor<1x7x7x1280xf32>) outs(%1 : tensor<1x1280xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %3 = addf %arg1, %arg2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x1280xf32>
  return %2 : tensor<1x1280xf32>
}

//      CHECK: func @reduce
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x7x7x1280xf32>)

//      CHECK: %[[FILL:.+]] = flow.dispatch.workgroups[{{.+}}]() : () -> tensor<1x1280xf32> =
// CHECK-NEXT:     (%{{.+}}: !flow.dispatch.tensor<writeonly:1x1280xf32>) {
//      CHECK:   linalg.init_tensor [1, 1280] : tensor<1x1280xf32>
// CHECK-NEXT:   linalg.fill

//      CHECK: %[[REDUCE:.+]] = flow.dispatch.workgroups[{{.+}}](%[[INPUT]], %[[FILL]]) : (tensor<1x7x7x1280xf32>, tensor<1x1280xf32>) -> %[[FILL]] =
// CHECK-NEXT:     (%[[ARG1:.+]]: !flow.dispatch.tensor<readonly:1x7x7x1280xf32>, %[[ARG2:.+]]: !flow.dispatch.tensor<readwrite:1x1280xf32>) {
//      CHECK:   %[[IN:.+]] = flow.dispatch.tensor.load %[[ARG1]]
//      CHECK:   %[[OUT:.+]] = flow.dispatch.tensor.load %[[ARG2]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:     ins(%[[IN]] : tensor<1x7x7x1280xf32>) outs(%[[OUT]] : tensor<1x1280xf32>)
//      CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[ARG2]]

//      CHECK: return %[[REDUCE]] : tensor<1x1280xf32>
