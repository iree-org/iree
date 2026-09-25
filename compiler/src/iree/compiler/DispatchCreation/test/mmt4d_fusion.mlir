// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{fuse-mmt4d=true}))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true fuse-mmt4d=true}))" %s | FileCheck %s --check-prefix=AGGRESSIVE

// The elementwise epilogue and the row-major result unpack join the mmt4d
// dispatch.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @mmt4d_relu_unpack(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<8x128x8x1xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<8x128x8x1xf32>
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<64x64xf32>)
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS]], %[[RHS]]
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D]] : tensor<8x8x8x8xf32>)
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU]]
// CHECK:           flow.return %[[UNPACK]] : tensor<64x64xf32>
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @mmt4d_relu_unpack(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<8x8x8x8xf32>) outs(%empty : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x8x8x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
  util.return %unpack : tensor<64x64xf32>
}

// -----

// Side operands of the epilogue may broadcast, such as a bias along N.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#bias = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-LABEL: util.func public @mmt4d_bias_relu_unpack(
// CHECK-SAME:    %[[BIAS:[a-zA-Z0-9]+]]: tensor<8x8xf32>
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<64x64xf32>)
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D]], %[[BIAS]] : tensor<8x8x8x8xf32>, tensor<8x8xf32>)
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU]]
// CHECK:           flow.return %[[UNPACK]] : tensor<64x64xf32>
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @mmt4d_bias_relu_unpack(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>, %bias: tensor<8x8xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #bias, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d, %bias : tensor<8x8x8x8xf32>, tensor<8x8xf32>) outs(%empty : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %add = arith.addf %in, %b : f32
    %max = arith.maximumf %add, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x8x8x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
  util.return %unpack : tensor<64x64xf32>
}

// -----

// Dynamic outer dimensions fuse like static ones.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @mmt4d_dynamic_relu_unpack(
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<?x64xf32>{%[[M]]})
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D]] : tensor<?x8x8x8xf32>)
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU]]
// CHECK:           flow.return %[[UNPACK]] : tensor<?x64xf32>
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @mmt4d_dynamic_relu_unpack(%lhs: tensor<?x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>, %m: index) -> tensor<?x64xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f32
  %m1 = tensor.dim %lhs, %c0 : tensor<?x128x8x1xf32>
  %empty = tensor.empty(%m1) : tensor<?x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x8x8x8xf32>) -> tensor<?x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<?x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<?x8x8x8xf32>) -> tensor<?x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<?x8x8x8xf32>) outs(%empty : tensor<?x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<?x8x8x8xf32>
  %dest = tensor.empty(%m) : tensor<?x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<?x8x8x8xf32> -> tensor<?x64xf32>
  util.return %unpack : tensor<?x64xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: util.func public @batch_mmt4d_relu_unpack(
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<2x64x64xf32>)
// CHECK:           %[[MMT4D:.+]] = linalg.batch_mmt4d
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D]] : tensor<2x8x8x8x8xf32>)
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU]]
// CHECK:           flow.return %[[UNPACK]] : tensor<2x64x64xf32>
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @batch_mmt4d_relu_unpack(%lhs: tensor<2x8x128x8x1xf32>, %rhs: tensor<2x8x128x8x1xf32>) -> tensor<2x64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x8x8x8x8xf32>) -> tensor<2x8x8x8x8xf32>
  %mmt4d = linalg.batch_mmt4d ins(%lhs, %rhs : tensor<2x8x128x8x1xf32>, tensor<2x8x128x8x1xf32>) outs(%fill : tensor<2x8x8x8x8xf32>) -> tensor<2x8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<2x8x8x8x8xf32>) outs(%empty : tensor<2x8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x8x8x8x8xf32>
  %dest = tensor.empty() : tensor<2x64x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %dest : tensor<2x8x8x8x8xf32> -> tensor<2x64x64xf32>
  util.return %unpack : tensor<2x64x64xf32>
}

// -----

// Chained elementwise epilogues, including a narrowing cast, join as well.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @mmt4d_relu_truncf_unpack(
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<64x64xbf16>)
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D]] : tensor<8x8x8x8xf32>)
// CHECK:           %[[TRUNCF:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[RELU]] : tensor<8x8x8x8xf32>)
// CHECK:             arith.truncf
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[TRUNCF]]
// CHECK:           flow.return %[[UNPACK]] : tensor<64x64xbf16>
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @mmt4d_relu_truncf_unpack(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>) -> tensor<64x64xbf16> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<8x8x8x8xf32>) outs(%empty : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x8x8x8xf32>
  %narrow_empty = tensor.empty() : tensor<8x8x8x8xbf16>
  %truncf = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%relu : tensor<8x8x8x8xf32>) outs(%narrow_empty : tensor<8x8x8x8xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %narrow = arith.truncf %in : f32 to bf16
    linalg.yield %narrow : bf16
  } -> tensor<8x8x8x8xbf16>
  %dest = tensor.empty() : tensor<64x64xbf16>
  %unpack = linalg.unpack %truncf outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xbf16> -> tensor<64x64xbf16>
  util.return %unpack : tensor<64x64xbf16>
}

// -----

// For narrow N, materialization transposes the result by swapping the last
// two entries of inner_dims_pos and outer_dims_perm. That unpack joins too.
// CHECK-LABEL: util.func public @mmt4d_narrow_n_transposed_unpack(
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<64x64xf32>)
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[MMT4D]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0]
// CHECK:           flow.return %[[UNPACK]] : tensor<64x64xf32>
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @mmt4d_narrow_n_transposed_unpack(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %mmt4d outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
  util.return %unpack : tensor<64x64xf32>
}

// -----

// Permuting only the outer dimensions is not a layout that materialization
// produces, so the unpack keeps its own dispatch. The epilogue still joins.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @no_fuse_permuted_unpack(
// CHECK:         %[[RELU_DISPATCH:.+]] = flow.dispatch.region -> (tensor<8x8x8x8xf32>)
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D]] : tensor<8x8x8x8xf32>)
// CHECK:           flow.return %[[RELU]]
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<64x64xf32>)
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU_DISPATCH]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
// CHECK:           flow.return %[[UNPACK]]
// CHECK:         util.return %[[RESULT]]
util.func public @no_fuse_permuted_unpack(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<8x8x8x8xf32>) outs(%empty : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x8x8x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
  util.return %unpack : tensor<64x64xf32>
}

// -----

// Dynamic inner tiles keep the unpack in its own dispatch.
// CHECK-LABEL: util.func public @no_fuse_dynamic_tile_unpack(
// CHECK:         %[[MMT4D_DISPATCH:.+]] = flow.dispatch.region -> (tensor<8x8x?x8xf32>{%{{.+}}})
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           flow.return %[[MMT4D]]
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[MMT4D_DISPATCH]]
// CHECK:           flow.return %[[UNPACK]]
// CHECK:         util.return %[[RESULT]]
util.func public @no_fuse_dynamic_tile_unpack(%lhs: tensor<8x128x?x1xf32>, %rhs: tensor<8x128x8x1xf32>, %tile: index) -> tensor<64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty(%tile) : tensor<8x8x?x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x?x8xf32>) -> tensor<8x8x?x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x?x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x?x8xf32>) -> tensor<8x8x?x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %mmt4d inner_dims_pos = [0, 1] inner_tiles = [%tile, 8] into %dest : tensor<8x8x?x8xf32> -> tensor<64x64xf32>
  util.return %unpack : tensor<64x64xf32>
}

// -----

// A transposing epilogue does not preserve the physical result layout, so it
// does not join the mmt4d dispatch, and neither does the unpack.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @no_fuse_transposed_epilogue(
// CHECK:         %[[MMT4D_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-NEXT:      flow.return %[[MMT4D]]
// CHECK:         %[[RELU_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D_DISPATCH]] : tensor<8x8x8x8xf32>)
// CHECK:           flow.return %[[RELU]]
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<64x64xf32>)
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU_DISPATCH]]
// CHECK-NEXT:      flow.return %[[UNPACK]]
util.func public @no_fuse_transposed_epilogue(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<8x8x8x8xf32>) outs(%empty : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x8x8x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
  util.return %unpack : tensor<64x64xf32>
}

// -----

// An epilogue that takes the mmt4d result as its init updates it in place, so
// it does not join the mmt4d dispatch.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @no_fuse_init_epilogue(
// CHECK:         %[[MMT4D_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-NEXT:      flow.return %[[MMT4D]]
// CHECK:         %[[ADD_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[ADD:.+]] = linalg.generic
// CHECK-SAME:        outs(%[[MMT4D_DISPATCH]] : tensor<8x8x8x8xf32>)
// CHECK:           flow.return %[[ADD]]
// CHECK:         util.return %[[ADD_DISPATCH]]
util.func public @no_fuse_init_epilogue(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>, %other: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %add = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%other : tensor<8x8x8x8xf32>) outs(%mmt4d : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<8x8x8x8xf32>
  util.return %add : tensor<8x8x8x8xf32>
}

// -----

// The epilogue reads the mmt4d result through an identity map but writes a
// permuted result, so it does not preserve the physical layout.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#perm = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
// CHECK-LABEL: util.func public @no_fuse_permuted_init_epilogue(
// CHECK:         %[[MMT4D_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-NEXT:      flow.return %[[MMT4D]]
// CHECK:         %[[RELU_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D_DISPATCH]] : tensor<8x8x8x8xf32>)
// CHECK:           flow.return %[[RELU]]
// CHECK:         %[[UNPACK_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU_DISPATCH]]
// CHECK:           flow.return %[[UNPACK]]
// CHECK:         util.return %[[UNPACK_DISPATCH]]
util.func public @no_fuse_permuted_init_epilogue(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>) -> tensor<64x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #perm], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<8x8x8x8xf32>) outs(%empty : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x8x8x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
  util.return %unpack : tensor<64x64xf32>
}

// -----

// The mmt4d result has a second use, so in the default mode nothing joins the
// mmt4d dispatch, and the unclaimed unpack still fuses with its own consumer.
// Aggressive fusion accepts multi-use roots and forms a single dispatch with two
// results, which ends at the unpack.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.func public @no_fuse_multi_use_result(
// CHECK:         %[[MMT4D_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-NEXT:      flow.return %[[MMT4D]]
// CHECK:         %[[RELU_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[RELU:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[MMT4D_DISPATCH]] : tensor<8x8x8x8xf32>)
// CHECK:           flow.return %[[RELU]]
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<64x64xf32>)
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[RELU_DISPATCH]]
// CHECK:           %[[NEG:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[UNPACK]] : tensor<64x64xf32>)
// CHECK:           flow.return %[[NEG]]
// CHECK:         util.return %[[RESULT]], %[[MMT4D_DISPATCH]]
// AGGRESSIVE-LABEL: util.func public @no_fuse_multi_use_result(
// AGGRESSIVE:         %[[DISPATCH:.+]]:2 = flow.dispatch.region -> (tensor<8x8x8x8xf32>, tensor<64x64xf32>)
// AGGRESSIVE:           %[[MMT4D:.+]] = linalg.mmt4d
// AGGRESSIVE:           %[[RELU:.+]] = linalg.generic
// AGGRESSIVE-SAME:        ins(%[[MMT4D]] : tensor<8x8x8x8xf32>)
// AGGRESSIVE:           %[[UNPACK:.+]] = linalg.unpack %[[RELU]]
// AGGRESSIVE:           flow.return %[[MMT4D]], %[[UNPACK]]
// AGGRESSIVE:         %[[NEG_DISPATCH:.+]] = flow.dispatch.region -> (tensor<64x64xf32>)
// AGGRESSIVE:           linalg.generic
// AGGRESSIVE-SAME:        ins(%[[DISPATCH]]#1 : tensor<64x64xf32>)
// AGGRESSIVE:         util.return %[[NEG_DISPATCH]], %[[DISPATCH]]#0
util.func public @no_fuse_multi_use_result(%lhs: tensor<8x128x8x1xf32>, %rhs: tensor<8x128x8x1xf32>) -> (tensor<64x64xf32>, tensor<8x8x8x8xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %mmt4d = linalg.mmt4d ins(%lhs, %rhs : tensor<8x128x8x1xf32>, tensor<8x128x8x1xf32>) outs(%fill : tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mmt4d : tensor<8x8x8x8xf32>) outs(%empty : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x8x8x8xf32>
  %dest = tensor.empty() : tensor<64x64xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<8x8x8x8xf32> -> tensor<64x64xf32>
  %neg = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%unpack : tensor<64x64xf32>) outs(%dest : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.negf %in : f32
    linalg.yield %0 : f32
  } -> tensor<64x64xf32>
  util.return %neg, %mmt4d : tensor<64x64xf32>, tensor<8x8x8x8xf32>
}
