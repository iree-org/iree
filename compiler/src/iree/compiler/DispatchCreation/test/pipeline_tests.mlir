// RUN: iree-opt --pass-pipeline="builtin.module(iree-dispatch-creation-fold-unit-extent-dims, iree-dispatch-creation-pipeline)" --split-input-file --mlir-print-local-scope %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
util.func public @main(%arg0: tensor<833xi32>, %arg1: tensor<833x833xf32>, %arg2: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant 5.66893432E-4 : f32
  %0 = tensor.empty() : tensor<833x833xf32>
  %1 = linalg.generic {
      indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %arg2 : tensor<833x833xf32>, tensor<f32>)
      outs(%0 : tensor<833x833xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %2 = arith.divf %b0, %b1 : f32
      linalg.yield %2 : f32
    } -> tensor<833x833xf32>
  %4 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map2], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg0, %1 : tensor<833xi32>, tensor<833xi32>, tensor<833x833xf32>)
      outs(%0 : tensor<833x833xf32>) {
    ^bb0(%b0 : i32, %b1 : i32, %b2 : f32, %b3 : f32):
      %5 = arith.cmpi eq, %b0, %b1 : i32
      %6 = arith.select %5, %b2, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<833x833xf32>
  %7 = tensor.empty() : tensor<f32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<f32>) -> tensor<f32>
  %9 = linalg.generic {
      indexing_maps = [#map2, #map3], iterator_types = ["reduction", "reduction"]}
      ins(%4 : tensor<833x833xf32>) outs(%7 : tensor<f32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %10 = arith.addf %b1, %b0 : f32
      linalg.yield %10 : f32
    } -> tensor<f32>
  util.return %9 : tensor<f32>
}
// Check that the linalg op with two reduction loops get folded into a single
// reduction which then prevents the parallel ops to be folded into it.
// See https://github.com/iree-org/iree/issues/13285
// CHECK-LABEL: func public @main
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<833xi32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<833x833xf32>
//  CHECK-SAME:     %[[ARG2:.+]]: tensor<f32>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups(%[[ARG0]], %[[ARG1]], %[[ARG2]])
//  CHECK-NEXT:       %[[ARG3:.+]]: !flow.dispatch.tensor<readonly:tensor<833xi32>>
//  CHECK-SAME:       %[[ARG4:.+]]: !flow.dispatch.tensor<readonly:tensor<833x833xf32>>
//  CHECK-SAME:       %[[ARG5:.+]]: !flow.dispatch.tensor<readonly:tensor<f32>>
//  CHECK-SAME:       %[[ARG6:.+]]: !flow.dispatch.tensor<writeonly:tensor<f32>>
//   CHECK-DAG:     %[[L0:.+]] = flow.dispatch.tensor.load %[[ARG3]]
//   CHECK-DAG:     %[[L1:.+]] = flow.dispatch.tensor.load %[[ARG4]]
//   CHECK-DAG:     %[[L2:.+]] = flow.dispatch.tensor.load %[[ARG5]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[L0]], %[[L0]], %[[L1]], %[[L2]] :
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC]], %[[ARG6]]
//       CHECK:   return %[[DISPATCH]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
util.func public @grouped_quantized_matmul(%arg0: tensor<4096x32x128xi4>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x1x4096xf32>
  %1 = tensor.empty() : tensor<4096x32x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi4>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %5 = arith.extui %in : i4 to i32
    %6 = arith.uitofp %5 : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<4096x32x128xf32>
  %4 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.mulf %in, %in_0 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1x1x4096xf32>
  util.return %4 : tensor<1x1x4096xf32>
}
// Check that the two linalg.generic ops are fused into the same dispatch.
// CHECK-LABEL: func public @grouped_quantized_matmul
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<4096x32x128xi4>,
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<1x1x32x128xf32>,
//  CHECK-SAME:     %[[ARG2:.+]]: tensor<4096x32x1xf32>,
//  CHECK-SAME:     %[[ARG3:.+]]: tensor<4096x32x1xf32>)
//   CHECK-DAG:   %[[RESHAPED_ARG2:.+]] = flow.tensor.reshape %[[ARG2]] : tensor<4096x32x1xf32> -> tensor<4096x32xf32>
//   CHECK-DAG:   %[[RESHAPED_ARG3:.+]] = flow.tensor.reshape %[[ARG3]] : tensor<4096x32x1xf32> -> tensor<4096x32xf32>
//   CHECK-DAG:   %[[RESHAPED_ARG1:.+]] = flow.tensor.reshape %[[ARG1]] : tensor<1x1x32x128xf32> -> tensor<32x128xf32>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups(%[[ARG0]], %[[RESHAPED_ARG2]], %[[RESHAPED_ARG3]], %[[RESHAPED_ARG1]])
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         iterator_types = ["parallel", "reduction", "reduction"]
//  CHECK-SAME:         ins(%{{.+}}, %[[GENERIC1]] :
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC2]]
//       CHECK:   %[[RESHAPE:.+]] = flow.tensor.reshape %[[DISPATCH]]
//       CHECK:   return %[[RESHAPE]]

// -----

util.func public @verify_operand_cse(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view {
  %c12 = arith.constant 12 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
  %2 = hal.tensor.import wait(%arg2) => %arg0 : !hal.buffer_view -> tensor<?x12x?x64xf32>{%0, %1}
  %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
  %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[3] : index
  %5 = hal.tensor.import wait(%arg2) => %arg1 : !hal.buffer_view -> tensor<?x12x64x?xf32>{%3, %4}
  %6 = arith.maxui %0, %3 : index
  %collapsed = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<?x12x?x64xf32> into tensor<?x?x64xf32>
  %collapsed_0 = tensor.collapse_shape %5 [[0, 1], [2], [3]] : tensor<?x12x64x?xf32> into tensor<?x64x?xf32>
  %7 = arith.muli %6, %c12 : index
  %8 = tensor.empty(%7, %1, %4) : tensor<?x?x?xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %10 = linalg.batch_matmul ins(%collapsed, %collapsed_0 : tensor<?x?x64xf32>, tensor<?x64x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %11 = arith.divui %7, %c12 : index
  %expanded = tensor.expand_shape %10 [[0, 1], [2], [3]] output_shape [%11, 12, %1, %4] : tensor<?x?x?xf32> into tensor<?x12x?x?xf32>
  %12 = hal.tensor.barrier join(%expanded : tensor<?x12x?x?xf32>) => %arg3 : !hal.fence
  %dim = tensor.dim %12, %c0 : tensor<?x12x?x?xf32>
  %dim_1 = tensor.dim %12, %c2 : tensor<?x12x?x?xf32>
  %dim_2 = tensor.dim %12, %c3 : tensor<?x12x?x?xf32>
  %13 = hal.tensor.export %12 : tensor<?x12x?x?xf32>{%dim, %dim_1, %dim_2} -> !hal.buffer_view
  util.return %13 : !hal.buffer_view
}
// Check that after forming dispatch.workgroup op the size of the
// `flow.tensor.load` and the dynamic dimension match. This is allows
// checking that the slice is a full slice. Running CSE before
// canonicalization makes this happen for this case.

// CHECK-LABEL: func public @verify_operand_cse
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
//   CHECK-DAG:     %[[DIM1:.+]] = flow.dispatch.workload.ordinal %{{.+}}, 0
//   CHECK-DAG:     %[[DIM2:.+]] = flow.dispatch.workload.ordinal %{{.+}}, 1
//   CHECK-DAG:     %[[DIM3:.+]] = flow.dispatch.workload.ordinal %{{.+}}, 2
//   CHECK-DAG:     %[[DIM4:.+]] = flow.dispatch.workload.ordinal %{{.+}}, 3
//       CHECK:   flow.dispatch.tensor.load
//  CHECK-SAME:       sizes = [%[[DIM1]], %[[DIM2]], 64]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x64xf32>>{%[[DIM1]], %[[DIM2]]}
//       CHECK:   flow.dispatch.tensor.load
//  CHECK-SAME:       sizes = [%[[DIM3]], 64, %[[DIM4]]]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x64x?xf32>>{%[[DIM3]], %[[DIM4]]}
