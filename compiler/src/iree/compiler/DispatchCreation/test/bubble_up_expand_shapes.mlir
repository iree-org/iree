// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-bubble-up-expand-shapes))" %s | FileCheck %s

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

#elementwise_trait = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel"]
}

// This test could actually fuse into 1 by using elementwise fusion. We could
// in reality, use all reductions with expansion on outer parallel loops also.
// Elementwise operations are just easier to write.
util.func public @diamond_propagate_expand_shape(%input : tensor<?x?xf16>)
                                                       -> tensor<2x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index

  %c1 = arith.constant 1.0 : f16
  %dim = tensor.dim %input, %c0 : tensor<?x?xf16>
  %empty = tensor.empty(%dim, %dim) : tensor<?x?xf16>

  %A = linalg.generic #elementwise_trait
  ins(%input : tensor<?x?xf16>) outs(%empty : tensor<?x?xf16>) {
  ^bb0(%in : f16, %out : f16):
    %add = arith.addf %in, %c1 : f16
    linalg.yield %add : f16
  } -> tensor<?x?xf16>

  %B = linalg.generic #elementwise_trait
  ins(%A : tensor<?x?xf16>) outs(%empty : tensor<?x?xf16>) {
  ^bb0(%in : f16, %out : f16):
    %add = arith.addf %in, %c1 : f16
    linalg.yield %add : f16
  } -> tensor<?x?xf16>

  %C = linalg.generic #elementwise_trait
  ins(%A : tensor<?x?xf16>) outs(%empty : tensor<?x?xf16>) {
  ^bb0(%in : f16, %out : f16):
    %add = arith.addf %in, %c1 : f16
    linalg.yield %add : f16
  } -> tensor<?x?xf16>

  // The canonical form would be to pass both inputs as ins, but for a consise
  // test, we pass it as outs so we can reuse the elementwise_trait.
  %D = linalg.generic #elementwise_trait
  ins(%B : tensor<?x?xf16>) outs(%C : tensor<?x?xf16>) {
  ^bb0(%in : f16, %out : f16):
    %add = arith.addf %in, %out : f16
    linalg.yield %add : f16
  } -> tensor<?x?xf16>

  %dimA = arith.divui %dim, %c2 : index
  %out = tensor.expand_shape %D [[0, 1], [2]] output_shape [2, %dimA, %dim] :
    tensor<?x?xf16> into tensor<2x?x?xf16>

  util.return %out : tensor<2x?x?xf16>
}

// Check that there is only 1 expand_shape at top
// CHECK-LABEL: diamond_propagate_expand_shape
// CHECK: tensor.expand_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: util.return

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
