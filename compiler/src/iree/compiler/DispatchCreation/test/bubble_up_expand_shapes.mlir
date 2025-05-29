// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-bubble-up-expand-shapes))" %s | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-bubble-up-expand-shapes{enable-bubble-up-expand-shapes-across-reduction-ops}))" %s | FileCheck %s --check-prefixes=CHECK,CHECK-AGGRESSIVE

util.func public @bubbble_expand_through_extract(%arg0 : tensor<2x4096x5120xf16>) -> (tensor<2x64x64x2560xf16>) {
  %extracted_slice_237 = tensor.extract_slice %arg0[0, 0, 0] [2, 4096, 2560] [1, 1, 1] : tensor<2x4096x5120xf16> to tensor<2x4096x2560xf16>
  %expanded_239 = tensor.expand_shape %extracted_slice_237 [[0], [1, 2], [3]] output_shape [2, 64, 64, 2560] : tensor<2x4096x2560xf16> into tensor<2x64x64x2560xf16>
  util.return %expanded_239 : tensor<2x64x64x2560xf16>
}

// CHECK-LABEL:  @bubbble_expand_through_extract
//       CHECK:    %[[EXPAND:.+]] = tensor.expand_shape
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice %[[EXPAND]]

// -----

util.func public @unsupported_bubbble_expand_through_extract(%arg0 : tensor<2x4096x5120xf16>) -> (tensor<2x32x64x2560xf16>) {
  %extracted_slice_237 = tensor.extract_slice %arg0[0, 0, 0] [2, 2048, 2560] [1, 1, 1] : tensor<2x4096x5120xf16> to tensor<2x2048x2560xf16>
  %expanded_239 = tensor.expand_shape %extracted_slice_237 [[0], [1, 2], [3]] output_shape [2, 32, 64, 2560] : tensor<2x2048x2560xf16> into tensor<2x32x64x2560xf16>
  util.return %expanded_239 : tensor<2x32x64x2560xf16>
}

// CHECK-LABEL:  @unsupported_bubbble_expand_through_extract
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:    %[[EXPAND:.+]] = tensor.expand_shape %[[EXTRACT]]

// -----

// Checks two things
// 1. Propagation of reshapes across attention operations
// 2. Use of folders to convert (expand(collapse)) -> (collapse)
util.func public @attention_v_reshape_propagation(%arg0: index,
    %arg1: tensor<4x8x4x128x?xf16>, %arg2: tensor<128x?x128xf16>,
    %arg3: tensor<128x?x128xf16>, %arg4: f16, %arg5: tensor<128x?x?xf16>)
    -> tensor<4x?x32x128xf16> {
  %0 = tensor.empty(%arg0) : tensor<4x?x32x128xf16>
  %1 = tensor.empty(%arg0) : tensor<128x?x128xf16>
  %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2], [3], [4]]
      : tensor<4x8x4x128x?xf16> into tensor<128x128x?xf16>
  %4 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]}
      ins(%arg2, %arg3, %collapsed, %arg4, %arg5
        : tensor<128x?x128xf16>, tensor<128x?x128xf16>, tensor<128x128x?xf16>,
          f16, tensor<128x?x?xf16>)
      outs(%1 : tensor<128x?x128xf16>) {
      ^bb0(%arg6: f32):
    iree_linalg_ext.yield %arg6 : f32
  } -> tensor<128x?x128xf16>
  %expanded = tensor.expand_shape %4 [[0, 1], [2], [3]]
      output_shape [4, 32, %arg0, 128]
      : tensor<128x?x128xf16> into tensor<4x32x?x128xf16>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<4x32x?x128xf16>) outs(%0 : tensor<4x?x32x128xf16>) {                                                                                                                                        ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x?x32x128xf16>
  util.return %5 : tensor<4x?x32x128xf16>
}
// CHECK-LABEL: func public @attention_v_reshape_propagation
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<4x8x4x128x?xf16>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}}, %[[ARG1]],
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ATTENTION]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[GENERIC]]
//       CHECK:   return %[[COLLAPSE]]

// -----

// Multiple uses of the producer

util.func @multiple_users(%arg0 : tensor<?x128xf16>,
    %arg1 : tensor<4x?x32x128xf16>) -> tensor<4x?x32x8x16xf16> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x128xf16>
  %empty = tensor.empty(%dim) : tensor<4x?x32x128xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<?x128xf16>)
      outs(%empty : tensor<4x?x32x128xf16>) {
    ^bb0(%b0: f16, %b1 : f16) :
      %iv0 = linalg.index 0 : index
      %iv1 = linalg.index 0 : index
      %iv2 = linalg.index 0 : index
      %iv3 = linalg.index 0 : index
      // This is not technically a gather, but doing this way to mimic
      // use case of rope computation in LLaMa
      %1 = tensor.extract %arg1[%iv0, %iv1, %iv2, %iv3] : tensor<4x?x32x128xf16>
      %2 = arith.addf %1, %b0 : f16
      linalg.yield %2 : f16
  } -> tensor<4x?x32x128xf16>
  %1 = tensor.dim %arg0, %c0 : tensor<?x128xf16>
  %2 = tensor.expand_shape %0 [[0], [1], [2], [3, 4]] output_shape [4, %1, 32, 8, 16]
      : tensor<4x?x32x128xf16> into tensor<4x?x32x8x16xf16>
  util.return %2 : tensor<4x?x32x8x16xf16>
}
// CHECK-LABEL: func public @multiple_users(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x128xf16>
//       CHECK:   %[[EXPAND_SHAPE:.+]] = tensor.expand_shape %[[ARG0]]
//       CHECK:   %[[EXPANDED_GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[EXPAND_SHAPE]] :
//       CHECK:   return %[[EXPANDED_GENERIC]]

// -----

// Bubbling through reductions should apply when enabled via flag.
util.func @bubble_up_through_reduction(%arg0: tensor<10x?xi64>) -> tensor<2x5xi64> {
  %outs = tensor.empty() : tensor<10xi64>
  %9 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<10x?xi64>) outs(%outs : tensor<10xi64>) {
  ^bb0(%in: i64, %out: i64):
    %x = arith.addi %in, %out : i64
    linalg.yield %x : i64
  } -> tensor<10xi64>
  %expanded = tensor.expand_shape %9 [[0, 1]] output_shape [2, 5] : tensor<10xi64> into tensor<2x5xi64>
  util.return %expanded : tensor<2x5xi64>
}
//      CHECK-LABEL: func public @bubble_up_through_reduction
//       CHECK-SAME:     %[[ARG0:.+]]: tensor
//    CHECK-DEFAULT:   %[[GENERIC:.+]] = linalg.generic {{.+}} ins(%[[ARG0]]
//    CHECK-DEFAULT:   %[[EXPANDED:.+]] = tensor.expand_shape %[[GENERIC]]
//    CHECK-DEFAULT:   return %[[EXPANDED]]

// CHECK-AGGRESSIVE:   %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-AGGRESSIVE:   %[[EXPANDED_GENERIC:.+]] = linalg.generic {{.+}} ins(%[[EXPANDED]]
// CHECK-AGGRESSIVE:   return %[[EXPANDED_GENERIC]]

// -----

// Check that dim resolution kicks in during expand shape propagation
util.func public @verify_dim_propagation(%arg0: index,
    %arg1: tensor<16x32x128x?xf8E4M3FNUZ>) -> tensor<4x4x32x128x?xbf16> {
  %c3 = arith.constant 3 : index
  %0 = tensor.empty(%arg0) : tensor<16x32x128x?xbf16>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<16x32x128x?xf8E4M3FNUZ>) outs(%0 : tensor<16x32x128x?xbf16>) {
    ^bb0(%in: f8E4M3FNUZ, %out: bf16):
      %2 = arith.extf %in : f8E4M3FNUZ to bf16
      linalg.yield %2 : bf16
    } -> tensor<16x32x128x?xbf16>
  %dim = tensor.dim %1, %c3 : tensor<16x32x128x?xbf16>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3], [4]]
      output_shape [4, 4, 32, 128, %dim]
      : tensor<16x32x128x?xbf16> into tensor<4x4x32x128x?xbf16>
  util.return %expanded : tensor<4x4x32x128x?xbf16>
}
// CHECK-LABEL: @verify_dim_propagation
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG1]]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[EXPAND]]
//       CHECK:   return %[[GENERIC]]

// -----

// Check if unit dim expansion in a cyclic expansion like graph could cause
// infinite behavior.
util.func public @test_no_infinite_loop_unit_dim_expansion(%arg0 : tensor<4xi64>, %arg1 : tensor<4xi64>, %arg3 : tensor<4xi64>) -> (tensor<4xi64>) {
  %c2_i64 = arith.constant 2 : i64
  %cst = arith.constant dense<[2, 1]> : tensor<2xi64>
  %c4 = arith.constant 4 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %__hoisted_tensor_4xi64 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %1 = tensor.empty() : tensor<4xi64>
  %9 = tensor.empty() : tensor<4xi64>
  %10:2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%9, %1 : tensor<4xi64>, tensor<4xi64>) {
  ^bb0(%out: i64, %out_0: i64):
    %16 = linalg.index 0 : index
    %17 = arith.remsi %16, %c4 : index
    %extracted = tensor.extract %arg0[%17] : tensor<4xi64>
    %extracted_1 = tensor.extract %arg1[%17] : tensor<4xi64>
    linalg.yield %extracted, %extracted_1 : i64, i64
  } -> (tensor<4xi64>, tensor<4xi64>)
  %expanded = tensor.expand_shape %10#0 [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "parallel"]}
    ins(%10#1, %expanded: tensor<4xi64>, tensor<4x1xi64>) outs(%1 : tensor<4xi64>) {
    ^bb0(%in: i64, %in0: i64, %out: i64):
      %idx = linalg.index 1 : index
      %cast = arith.index_cast %idx : index to i64
      %add = arith.addi %in, %in0: i64
      %add1 = arith.addi %add, %cast: i64
      linalg.yield %add1 : i64
    } -> tensor<4xi64>

  util.return %11 : tensor<4xi64>
}

// CHECK-LABEL: test_no_infinite_loop_unit_dim_expansion
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape

// -----

util.func @dont_propagate_edge_unit_reshapes(%arg0: tensor<?x1xi32>) -> tensor<?xi32> {
  %collapsed = tensor.collapse_shape %arg0[[0, 1]] : tensor<?x1xi32> into tensor<?xi32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed: tensor<?xi32>) outs(%collapsed: tensor<?xi32>){
^bb0(%in : i32, %out : i32):
   %1 = arith.addi %in, %in : i32
  linalg.yield %1 : i32
  } -> tensor<?xi32>
  util.return %0 : tensor<?xi32>
}
// CHECK-LABEL: util.func public @dont_propagate_edge_unit_reshapes
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]+]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]]
//       CHECK:   %[[VAL:.+]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//       CHECK:   util.return %[[VAL]] : tensor<?xi32>
