// RUN: iree-opt --pass-pipeline="builtin.module(iree-preprocessing-attr-based-pipeline, iree-dispatch-creation-fold-unit-extent-dims, iree-dispatch-creation-pipeline)" --split-input-file --mlir-print-local-scope %s | FileCheck %s

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
//  CHECK-NEXT:       %[[ARG3:.+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<833xi32>>
//  CHECK-SAME:       %[[ARG4:.+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<833x833xf32>>
//  CHECK-SAME:       %[[ARG5:.+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<f32>>
//  CHECK-SAME:       %[[ARG6:.+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
//   CHECK-DAG:     %[[L0:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG3]]
//   CHECK-DAG:     %[[L1:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG4]]
//   CHECK-DAG:     %[[L2:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG5]]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[L0]], %[[L0]], %[[L1]], %[[L2]] :
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[ARG6]]
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
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[GENERIC2]]
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
//   CHECK-DAG:     %[[DIM1:.+]] = iree_tensor_ext.dispatch.workload.ordinal %{{.+}}, 0
//   CHECK-DAG:     %[[DIM2:.+]] = iree_tensor_ext.dispatch.workload.ordinal %{{.+}}, 1
//   CHECK-DAG:     %[[DIM3:.+]] = iree_tensor_ext.dispatch.workload.ordinal %{{.+}}, 2
//   CHECK-DAG:     %[[DIM4:.+]] = iree_tensor_ext.dispatch.workload.ordinal %{{.+}}, 3
//       CHECK:   iree_tensor_ext.dispatch.tensor.load
//  CHECK-SAME:       sizes = [%[[DIM1]], %[[DIM2]], 64]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x64xf32>>{%[[DIM1]], %[[DIM2]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.load
//  CHECK-SAME:       sizes = [%[[DIM3]], 64, %[[DIM4]]]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x64x?xf32>>{%[[DIM3]], %[[DIM4]]}

// -----

util.func public @attention_rope_fusion(%arg0: index, %arg1: tensor<?x128xf32>,
    %arg2: tensor<4x8x4x?x128xf16>, %arg3: tensor<4x8x4x128x?xf16>, %arg4: f16,
    %arg5: tensor<4x8x4x?x?xf16>, %arg6: tensor<4x?x32x128xf16>)
    -> tensor<4x8x4x?x128xf16> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty(%arg0) : tensor<4x32x?x128xf16>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<?x128xf32>) outs(%0 : tensor<4x32x?x128xf16>) {
  ^bb0(%in: f32, %out: f16):
    %6 = linalg.index 0 : index
    %7 = linalg.index 1 : index
    %8 = linalg.index 2 : index
    %9 = linalg.index 3 : index
    %10 = arith.divui %9, %c2 : index
    %11 = arith.remui %9, %c2 : index
    %12 = math.cos %in : f32
    %13 = math.sin %in : f32
    %14 = arith.muli %10, %c2 : index
    %15 = arith.addi %14, %c1 : index
    %extracted = tensor.extract %arg6[%6, %7, %8, %14] : tensor<4x?x32x128xf16>
    %16 = arith.extf %extracted : f16 to f32
    %extracted_0 = tensor.extract %arg6[%6, %7, %8, %15] : tensor<4x?x32x128xf16>
    %17 = arith.extf %extracted_0 : f16 to f32
    %18 = arith.cmpi eq, %11, %c0 : index
    %19 = arith.mulf %16, %12 : f32
    %20 = arith.mulf %17, %13 : f32
    %21 = arith.subf %19, %20 : f32
    %22 = arith.mulf %17, %12 : f32
    %23 = arith.mulf %16, %13 : f32
    %24 = arith.addf %22, %23 : f32
    %25 = arith.select %18, %21, %24 : f32
    %26 = arith.truncf %25 : f32 to f16
    linalg.yield %26 : f16
  } -> tensor<4x32x?x128xf16>
  %expanded = tensor.expand_shape %1 [[0], [1, 2], [3], [4]]
    output_shape [4, 8, 4, %arg0, 128]
        : tensor<4x32x?x128xf16> into tensor<4x8x4x?x128xf16>
  %2 = tensor.empty(%arg0) : tensor<4x?x8x4x128xf16>
  %3 = tensor.empty(%arg0) : tensor<4x8x4x?x128xf16>
  %4 = iree_linalg_ext.attention {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>]}
      ins(%expanded, %arg2, %arg3, %arg4, %arg5
        : tensor<4x8x4x?x128xf16>, tensor<4x8x4x?x128xf16>,
          tensor<4x8x4x128x?xf16>, f16, tensor<4x8x4x?x?xf16>)
      outs(%3 : tensor<4x8x4x?x128xf16>) {
    ^bb0(%arg7: f32):
      iree_linalg_ext.yield %arg7 : f32
  } -> tensor<4x8x4x?x128xf16>
  util.return %4 : tensor<4x8x4x?x128xf16>
}
// CHECK-LABEL: util.func public @attention_rope_fusion
//   CHECK-NOT:   linalg.generic
//       CHECK:   flow.dispatch.workgroup
//       CHECK:     %[[GATHER:.+]] = linalg.generic
//       CHECK:   flow.dispatch.workgroup
//       CHECK:     %[[ATTENTION:.+]] = iree_linalg_ext.attention

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @verify_bubbling(%arg0: !hal.buffer_view, %arg2: !hal.fence) -> !hal.buffer_view {
  %c12 = arith.constant 12 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f32
  %2 = hal.tensor.import wait(%arg2) => %arg0 : !hal.buffer_view -> tensor<10x10xf32>
  %empty = tensor.empty() : tensor<10x10xf32>
  %elementwise = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<10x10xf32>) outs(%empty : tensor<10x10xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
  } -> tensor<10x10xf32>
  %extract = tensor.extract_slice %elementwise[0, 0][4, 10][1, 1] : tensor<10x10xf32> to tensor<4x10xf32>
  %expanded = tensor.expand_shape %extract [[0, 1], [2, 3]] output_shape[2, 2, 2, 5] : tensor<4x10xf32> into tensor<2x2x2x5xf32>
  %extract2 = tensor.extract_slice %expanded[0, 0, 0, 0][1, 1, 2, 5][1, 1, 1, 1] : tensor<2x2x2x5xf32> to tensor<2x5xf32>
  %empty2 = tensor.empty() : tensor<2x5xf32>
  %elementwise2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%extract2 : tensor<2x5xf32>) outs(%empty2 : tensor<2x5xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
  } -> tensor<2x5xf32>
  %12 = hal.tensor.barrier join(%elementwise2 : tensor<2x5xf32>) => %arg2 : !hal.fence
  %13 = hal.tensor.export %12 : tensor<2x5xf32> -> !hal.buffer_view
  util.return %13 : !hal.buffer_view
}

// Check that all `tensor.expand_shape` and `tensor.extract_slice` ops get bubbled
// and allow for fusion of the 2 elementwise ops. This should result in a single
// dispatch.

// CHECK-LABEL: util.func public @verify_bubbling
//   CHECK-NOT:   linalg.generic
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroup
//       CHECK:     %[[GEN:.+]] = linalg.generic
//       CHECK:      iree_tensor_ext.dispatch.tensor.store %[[GEN]]
//   CHECK-NOT:   linalg.generic

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @set_encoding_op(%arg0 : tensor<?x?xf32>)
    -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0
      : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  util.return %0 : tensor<?x?xf32, #encoding>
}
// CHECK-LABEL: util.func public @set_encoding_op
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[D0:.+]] = tensor.dim %[[SRC]], %[[C0]]
// CHECK-DAG:     %[[D1:.+]] = tensor.dim %[[SRC]], %[[C1]]
// CHECK:         %[[RES:.+]] = flow.tensor.encode %[[SRC]] : tensor<?x?xf32>{%[[D0]], %[[D1]]} -> tensor<?x?xf32, #iree_encoding.testing_encoding<>>{%[[D0]], %[[D1]]}
// CHECK:         util.return %[[RES]]

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @unset_encoding_op(%arg0 : tensor<?x?xf32, #encoding>, %d0: index, %d1: index)
    -> tensor<?x?xf32> {
  %0 = iree_encoding.unset_encoding %arg0
      : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @unset_encoding_op
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[D1:[a-zA-Z0-9]+]]
// CHECK:         %[[RES:.+]] = flow.tensor.encode %[[SRC]] : tensor<?x?xf32, #iree_encoding.testing_encoding<>>{%[[D0]], %[[D1]]} -> tensor<?x?xf32>{%[[D0]], %[[D1]]}
// CHECK:         util.return %[[RES]]

// -----

// Check that we are able to collapse in presence of unit dims in the make single dispatch pipeline

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @make_single_dispatch(%arg0: tensor<16x8x32x2048xbf16>, %arg1: tensor<16x8x32x4096xbf16>) -> tensor<4096x1x1x2048xf32>
 attributes {preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">} {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<4096x1x1x2048xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4096x1x1x2048xf32>) -> tensor<4096x1x1x2048xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x8x32x2048xbf16>, tensor<16x8x32x4096xbf16>) outs(%3 : tensor<4096x1x1x2048xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %9 = arith.extf %in : bf16 to f32
      %10 = arith.extf %in_0 : bf16 to f32
      %11 = arith.mulf %9, %10 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<4096x1x1x2048xf32>
    util.return %4 : tensor<4096x1x1x2048xf32>
  }

// CHECK-LABEL: util.func public @make_single_dispatch
//       CHECK: linalg.generic
//  CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4096x2048xbf16>, tensor<4096x4096xbf16>)
//  CHECK-SAME: outs(%{{.*}} :  tensor<4096x2048xf32>)

// -----

util.func public @gather_matmul(%source : tensor<20x20x100xi32>, %indices : tensor<100x2xi32>, %arg2 : tensor<100x100xi32>, %arg3 : tensor<100x100xi32>) -> tensor<100x100xi32> {
  %empty = tensor.empty() : tensor<100x100xi32>
  %result = iree_linalg_ext.gather dimension_map = [1, 0]
                          ins(%source, %indices : tensor<20x20x100xi32>, tensor<100x2xi32>)
                          outs(%empty: tensor<100x100xi32>) -> tensor<100x100xi32>
  %mm = linalg.matmul_transpose_b ins(%result, %arg2 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%arg3 : tensor<100x100xi32>) -> tensor<100x100xi32>
  util.return %mm : tensor<100x100xi32>
}
// CHECK-LABEL: util.func public @gather_matmul
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[GATHER:.+]] = iree_linalg_ext.gather
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul_transpose_b
//  CHECK-SAME:       ins(%[[GATHER]]
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[MATMUL]]
//       CHECK:   util.return %[[DISPATCH]]

// -----

util.func public @single_gather(%source : tensor<20x20x100xi32>, %indices : tensor<100x2xi32>, %arg2 : tensor<100x100xi32>, %arg3 : tensor<100x100xi32>) -> tensor<100x100xi32> {
  %empty = tensor.empty() : tensor<100x100xi32>
  %result = iree_linalg_ext.gather dimension_map = [1, 0]
                          ins(%source, %indices : tensor<20x20x100xi32>, tensor<100x2xi32>)
                          outs(%empty: tensor<100x100xi32>) -> tensor<100x100xi32>
  util.return %result : tensor<100x100xi32>
}
// Make sure single gather gets wrapped in dispatch.
// CHECK-LABEL: util.func public @single_gather
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[GATHER:.+]] = iree_linalg_ext.gather
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[GATHER]]
//       CHECK:   util.return %[[DISPATCH]]
