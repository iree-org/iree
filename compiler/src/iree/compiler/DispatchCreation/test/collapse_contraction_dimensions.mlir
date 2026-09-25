// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-collapse-contraction-dimensions))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-collapse-contraction-dimensions,iree-dispatch-creation-annotate-data-tiling-hints,iree-dispatch-creation-set-encoding))' %s | FileCheck %s --check-prefix=ENCODE

// Two contiguous M dimensions collapse into one, and the result is expanded
// back to the original shape. Encoding annotation rejects the uncollapsed form,
// so the ENCODE run checks that the collapsed contraction now gets encodings.
// CHECK-LABEL: @multi_m(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<2x3x8xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<8x4xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<2x3x4xf32>
// CHECK-DAG:     %[[A_2D:.+]] = tensor.collapse_shape %[[A]] {{\[}}[0, 1], [2]] : tensor<2x3x8xf32> into tensor<6x8xf32>
// CHECK-DAG:     %[[INIT_2D:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1], [2]] : tensor<2x3x4xf32> into tensor<6x4xf32>
// CHECK:         %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:      ins(%[[A_2D]], %[[B]] : tensor<6x8xf32>, tensor<8x4xf32>)
// CHECK-SAME:      outs(%[[INIT_2D]] : tensor<6x4xf32>)
// CHECK:         %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]] output_shape [2, 3, 4] : tensor<6x4xf32> into tensor<2x3x4xf32>
// CHECK:         util.return %[[RESULT]]
// ENCODE-LABEL: @multi_m(
// ENCODE:         %[[A_2D:.+]] = tensor.collapse_shape
// ENCODE:         %[[LHS:.+]] = iree_encoding.set_encoding %[[A_2D]] : tensor<6x8xf32> -> tensor<6x8xf32, #
// ENCODE:         %[[MATMUL:.+]] = linalg.generic
// ENCODE-SAME:      ins(%[[LHS]],
// ENCODE:         %[[UNSET:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
// ENCODE:         tensor.expand_shape %[[UNSET]]
util.func public @multi_m(%a: tensor<2x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x3x8xf32>, tensor<8x4xf32>)
    outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}

// -----

// Only the K dimensions collapse; the result keeps its shape, so no expand is
// needed.
// CHECK-LABEL: @multi_k(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<256x64x2xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<64x2x512xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<256x512xf32>
// CHECK-DAG:     %[[A_2D:.+]] = tensor.collapse_shape %[[A]] {{\[}}[0], [1, 2]] : tensor<256x64x2xf32> into tensor<256x128xf32>
// CHECK-DAG:     %[[B_2D:.+]] = tensor.collapse_shape %[[B]] {{\[}}[0, 1], [2]] : tensor<64x2x512xf32> into tensor<128x512xf32>
// CHECK:         %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:      ins(%[[A_2D]], %[[B_2D]] : tensor<256x128xf32>, tensor<128x512xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<256x512xf32>)
// CHECK-NOT:     tensor.expand_shape
// CHECK:         util.return %[[MATMUL]]
util.func public @multi_k(%a: tensor<256x64x2xf32>, %b: tensor<64x2x512xf32>,
                          %init: tensor<256x512xf32>) -> tensor<256x512xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n, k0, k1) -> (m, k0, k1)>,
                     affine_map<(m, n, k0, k1) -> (k0, k1, n)>,
                     affine_map<(m, n, k0, k1) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%a, %b : tensor<256x64x2xf32>, tensor<64x2x512xf32>)
    outs(%init : tensor<256x512xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<256x512xf32>
  util.return %result : tensor<256x512xf32>
}

// -----

// Two batch dimensions collapse across all three operands.
// CHECK-LABEL: @multi_batch(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<4x8x256x128xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<4x8x128x512xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<4x8x256x512xf32>
// CHECK-DAG:     %[[A_3D:.+]] = tensor.collapse_shape %[[A]] {{\[}}[0, 1], [2], [3]] : tensor<4x8x256x128xf32> into tensor<32x256x128xf32>
// CHECK-DAG:     %[[B_3D:.+]] = tensor.collapse_shape %[[B]] {{\[}}[0, 1], [2], [3]] : tensor<4x8x128x512xf32> into tensor<32x128x512xf32>
// CHECK-DAG:     %[[INIT_3D:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1], [2], [3]] : tensor<4x8x256x512xf32> into tensor<32x256x512xf32>
// CHECK:         %[[BATCH_MATMUL:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:      ins(%[[A_3D]], %[[B_3D]] : tensor<32x256x128xf32>, tensor<32x128x512xf32>)
// CHECK-SAME:      outs(%[[INIT_3D]] : tensor<32x256x512xf32>)
// CHECK:         %[[RESULT:.+]] = tensor.expand_shape %[[BATCH_MATMUL]] {{\[}}[0, 1], [2], [3]] output_shape [4, 8, 256, 512]
// CHECK:         util.return %[[RESULT]]
util.func public @multi_batch(%a: tensor<4x8x256x128xf32>, %b: tensor<4x8x128x512xf32>,
                              %init: tensor<4x8x256x512xf32>) -> tensor<4x8x256x512xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(b0, b1, m, n, k) -> (b0, b1, m, k)>,
                     affine_map<(b0, b1, m, n, k) -> (b0, b1, k, n)>,
                     affine_map<(b0, b1, m, n, k) -> (b0, b1, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<4x8x256x128xf32>, tensor<4x8x128x512xf32>)
    outs(%init : tensor<4x8x256x512xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<4x8x256x512xf32>
  util.return %result : tensor<4x8x256x512xf32>
}

// -----

// A batch dimension that only the LHS and result carry is an M dimension; it
// collapses with the other M dimension. Extending bodies are preserved.
// CHECK-LABEL: @broadcast_rhs_batch_mmt(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<16x1024x1280xi8>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<10240x1280xi8>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<16x1024x10240xi32>
// CHECK-DAG:     %[[A_2D:.+]] = tensor.collapse_shape %[[A]] {{\[}}[0, 1], [2]] : tensor<16x1024x1280xi8> into tensor<16384x1280xi8>
// CHECK-DAG:     %[[INIT_2D:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1], [2]] : tensor<16x1024x10240xi32> into tensor<16384x10240xi32>
// CHECK:         %[[MMT:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[A_2D]], %[[B]] : tensor<16384x1280xi8>, tensor<10240x1280xi8>)
// CHECK-SAME:      outs(%[[INIT_2D]] : tensor<16384x10240xi32>)
// CHECK:           arith.extsi
// CHECK:           arith.extsi
// CHECK:           arith.muli
// CHECK:           arith.addi
// CHECK:         %[[RESULT:.+]] = tensor.expand_shape %[[MMT]] {{\[}}[0, 1], [2]] output_shape [16, 1024, 10240]
// CHECK:         util.return %[[RESULT]]
util.func public @broadcast_rhs_batch_mmt(%a: tensor<16x1024x1280xi8>, %b: tensor<10240x1280xi8>,
                                          %init: tensor<16x1024x10240xi32>) -> tensor<16x1024x10240xi32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                     affine_map<(m0, m1, n, k) -> (n, k)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<16x1024x1280xi8>, tensor<10240x1280xi8>)
    outs(%init : tensor<16x1024x10240xi32>) {
  ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
    %lhs_ext = arith.extsi %lhs : i8 to i32
    %rhs_ext = arith.extsi %rhs : i8 to i32
    %mul = arith.muli %lhs_ext, %rhs_ext : i32
    %sum = arith.addi %acc, %mul : i32
    linalg.yield %sum : i32
  } -> tensor<16x1024x10240xi32>
  util.return %result : tensor<16x1024x10240xi32>
}

// -----

// A dynamic outer M dimension is recovered from the input for the expand.
// CHECK-LABEL: @dynamic_multi_m(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<?x3x8xf32>
// CHECK:         %[[M0:.+]] = tensor.dim %[[A]]
// CHECK:         %[[A_2D:.+]] = tensor.collapse_shape %[[A]] {{\[}}[0, 1], [2]] : tensor<?x3x8xf32> into tensor<?x8xf32>
// CHECK:         %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[A_2D]],
// CHECK:         %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]] output_shape [%[[M0]], 3, 4] : tensor<?x4xf32> into tensor<?x3x4xf32>
// CHECK:         util.return %[[RESULT]]
util.func public @dynamic_multi_m(%a: tensor<?x3x8xf32>, %b: tensor<8x4xf32>, %init: tensor<?x3x4xf32>) -> tensor<?x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, m1, k)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<?x3x8xf32>, tensor<8x4xf32>)
    outs(%init : tensor<?x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<?x3x4xf32>
  util.return %result : tensor<?x3x4xf32>
}

// -----

// The M dimensions are not contiguous in the LHS, so nothing collapses.
// CHECK-LABEL: @no_collapse_noncontiguous_m(
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-NOT:     tensor.expand_shape
// CHECK:         util.return
util.func public @no_collapse_noncontiguous_m(%a: tensor<2x8x3xf32>, %b: tensor<8x4xf32>, %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(m0, m1, n, k) -> (m0, k, m1)>,
                     affine_map<(m0, m1, n, k) -> (k, n)>,
                     affine_map<(m0, m1, n, k) -> (m0, m1, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x8x3xf32>, tensor<8x4xf32>)
    outs(%init : tensor<2x3x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x3x4xf32>
  util.return %result : tensor<2x3x4xf32>
}
