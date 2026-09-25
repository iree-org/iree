// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions))' %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{fuse-data-tiled-convolution=true}))' %s | FileCheck %s --check-prefixes=CHECK,FUSED
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true fuse-data-tiled-convolution=true}))' %s | FileCheck %s --check-prefix=AGGRESSIVE

// The data-tiled convolution produces [N, OC/k0, OH, OW, k0]. Unpacking the k0
// tile into NHWC joins the convolution dispatch only when the option is set.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
// DEFAULT-LABEL: util.func public @nhwc_unpack(
// DEFAULT:         %[[CONV_DISPATCH:.+]] = flow.dispatch.region
// DEFAULT:           %[[CONV:.+]] = linalg.generic
// DEFAULT:           flow.return %[[CONV]]
// DEFAULT:         flow.dispatch.region
// DEFAULT:           linalg.unpack %[[CONV_DISPATCH]]
// FUSED-LABEL: util.func public @nhwc_unpack(
// FUSED:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<2x14x14x16xf32>)
// FUSED:           %[[CONV:.+]] = linalg.generic
// FUSED:           %[[UNPACK:.+]] = linalg.unpack %[[CONV]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3]
// FUSED:           flow.return %[[UNPACK]]
// FUSED-NOT:     flow.dispatch.region
// FUSED:         util.return %[[RESULT]]
util.func public @nhwc_unpack(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> tensor<2x14x14x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %dest = tensor.empty() : tensor<2x14x14x16xf32>
  %unpack = linalg.unpack %conv outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dest : tensor<2x2x14x14x8xf32> -> tensor<2x14x14x16xf32>
  util.return %unpack : tensor<2x14x14x16xf32>
}

// -----

// Unpacking into NCHW keeps the outer dimension order.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
// FUSED-LABEL: util.func public @nchw_unpack(
// FUSED:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<2x16x14x14xf32>)
// FUSED:           %[[CONV:.+]] = linalg.generic
// FUSED:           %[[UNPACK:.+]] = linalg.unpack %[[CONV]] inner_dims_pos = [1]
// FUSED:           flow.return %[[UNPACK]]
// FUSED-NOT:     flow.dispatch.region
// FUSED:         util.return %[[RESULT]]
util.func public @nchw_unpack(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> tensor<2x16x14x14xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %dest = tensor.empty() : tensor<2x16x14x14xf32>
  %unpack = linalg.unpack %conv inner_dims_pos = [1] inner_tiles = [8] into %dest : tensor<2x2x14x14x8xf32> -> tensor<2x16x14x14xf32>
  util.return %unpack : tensor<2x16x14x14xf32>
}

// -----

// A batchless convolution arrives with its unit batch dimension collapsed into
// the outer channel dimension. The collapse and the HWC unpack join together
// with an elementwise epilogue.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
#identity = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// FUSED-LABEL: util.func public @unit_batch_relu_hwc_unpack(
// FUSED:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<14x14x16xf32>)
// FUSED:           %[[CONV:.+]] = linalg.generic
// FUSED:           %[[RELU:.+]] = linalg.generic
// FUSED-SAME:        ins(%[[CONV]] : tensor<1x2x14x14x8xf32>)
// FUSED:           %[[COLLAPSED:.+]] = tensor.collapse_shape %[[RELU]] {{\[}}[0, 1], [2], [3], [4]]
// FUSED:           %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [2, 0, 1] inner_dims_pos = [2]
// FUSED:           flow.return %[[UNPACK]]
// FUSED-NOT:     flow.dispatch.region
// FUSED:         util.return %[[RESULT]]
util.func public @unit_batch_relu_hwc_unpack(%input: tensor<1x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> tensor<14x14x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x14x14x8xf32>) -> tensor<1x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<1x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<1x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<1x2x14x14x8xf32>
  %relu = linalg.generic {indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%conv : tensor<1x2x14x14x8xf32>) outs(%empty : tensor<1x2x14x14x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x2x14x14x8xf32>
  %collapsed = tensor.collapse_shape %relu [[0, 1], [2], [3], [4]] : tensor<1x2x14x14x8xf32> into tensor<2x14x14x8xf32>
  %dest = tensor.empty() : tensor<14x14x16xf32>
  %unpack = linalg.unpack %collapsed outer_dims_perm = [2, 0, 1] inner_dims_pos = [2] inner_tiles = [8] into %dest : tensor<2x14x14x8xf32> -> tensor<14x14x16xf32>
  util.return %unpack : tensor<14x14x16xf32>
}

// -----

// Permuting the batch dimension is not a layout that materialization produces.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
// FUSED-LABEL: util.func public @no_fuse_permuted_unpack(
// FUSED:         %[[CONV_DISPATCH:.+]] = flow.dispatch.region
// FUSED:           %[[CONV:.+]] = linalg.generic
// FUSED:           flow.return %[[CONV]]
// FUSED:         flow.dispatch.region
// FUSED:           linalg.unpack %[[CONV_DISPATCH]]
util.func public @no_fuse_permuted_unpack(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> tensor<2x14x14x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %dest = tensor.empty() : tensor<2x14x14x16xf32>
  %unpack = linalg.unpack %conv outer_dims_perm = [3, 0, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dest : tensor<2x2x14x14x8xf32> -> tensor<2x14x14x16xf32>
  util.return %unpack : tensor<2x14x14x16xf32>
}

// -----

// Collapsing a non-unit batch dimension mixes batch and channel data.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
// FUSED-LABEL: util.func public @no_fuse_nonunit_batch_collapse(
// FUSED:         %[[CONV_DISPATCH:.+]] = flow.dispatch.region
// FUSED:           %[[CONV:.+]] = linalg.generic
// FUSED:           flow.return %[[CONV]]
// FUSED:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[CONV_DISPATCH]]
// FUSED:         flow.dispatch.region
// FUSED:           linalg.unpack %[[COLLAPSED]]
util.func public @no_fuse_nonunit_batch_collapse(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> tensor<14x14x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3], [4]] : tensor<2x2x14x14x8xf32> into tensor<4x14x14x8xf32>
  %dest = tensor.empty() : tensor<14x14x32xf32>
  %unpack = linalg.unpack %collapsed outer_dims_perm = [2, 0, 1] inner_dims_pos = [2] inner_tiles = [8] into %dest : tensor<4x14x14x8xf32> -> tensor<14x14x32xf32>
  util.return %unpack : tensor<14x14x32xf32>
}

// -----

// Batchless CHW is not matched. The unit-batch collapse only joins together
// with an unpack that joins, so neither of them is fused.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
// FUSED-LABEL: util.func public @no_fuse_unit_batch_chw_unpack(
// FUSED:         %[[CONV_DISPATCH:.+]] = flow.dispatch.region
// FUSED:           %[[CONV:.+]] = linalg.generic
// FUSED:           flow.return %[[CONV]]
// FUSED:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[CONV_DISPATCH]]
// FUSED:         flow.dispatch.region
// FUSED:           linalg.unpack %[[COLLAPSED]]
util.func public @no_fuse_unit_batch_chw_unpack(%input: tensor<1x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> tensor<16x14x14xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x14x14x8xf32>) -> tensor<1x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<1x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<1x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<1x2x14x14x8xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3], [4]] : tensor<1x2x14x14x8xf32> into tensor<2x14x14x8xf32>
  %dest = tensor.empty() : tensor<16x14x14xf32>
  %unpack = linalg.unpack %collapsed inner_dims_pos = [0] inner_tiles = [8] into %dest : tensor<2x14x14x8xf32> -> tensor<16x14x14xf32>
  util.return %unpack : tensor<16x14x14xf32>
}

// -----

// Consumers that do not preserve the data-tiled layout fuse with the
// convolution as they do without the option, which only adds the result
// unpack.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
#id = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#swap_hw = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
// CHECK-LABEL: util.func public @permuted_consumer(
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<2x2x14x14x8xf32>)
// CHECK:           %[[CONV:.+]] = linalg.generic
// CHECK:           %[[NEG:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[CONV]] : tensor<2x2x14x14x8xf32>)
// CHECK:           flow.return %[[NEG]]
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @permuted_consumer(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> tensor<2x2x14x14x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %neg = linalg.generic {indexing_maps = [#swap_hw, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%conv : tensor<2x2x14x14x8xf32>) outs(%empty : tensor<2x2x14x14x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.negf %in : f32
    linalg.yield %0 : f32
  } -> tensor<2x2x14x14x8xf32>
  util.return %neg : tensor<2x2x14x14x8xf32>
}

// -----

// The same holds for a consumer that updates the convolution result in place.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
#id = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: util.func public @init_consumer(
// CHECK:         %[[RESULT:.+]] = flow.dispatch.region -> (tensor<2x2x14x14x8xf32>)
// CHECK:           %[[CONV:.+]] = linalg.generic
// CHECK:           %[[ADD:.+]] = linalg.generic
// CHECK-SAME:        outs(%[[CONV]] : tensor<2x2x14x14x8xf32>)
// CHECK:           flow.return %[[ADD]]
// CHECK-NOT:     flow.dispatch.region
// CHECK:         util.return %[[RESULT]]
util.func public @init_consumer(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>, %other: tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %add = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%other : tensor<2x2x14x14x8xf32>) outs(%conv : tensor<2x2x14x14x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<2x2x14x14x8xf32>
  util.return %add : tensor<2x2x14x14x8xf32>
}

// -----

// The collapse has a second use, so the unpack could not follow it into the
// convolution dispatch, and the dispatch would end in the collapse. Neither
// joins the dispatch.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
// CHECK-LABEL: util.func public @no_fuse_multi_use_collapse(
// CHECK:         %[[CONV_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[CONV:.+]] = linalg.generic
// CHECK:           flow.return %[[CONV]]
// CHECK:         %[[COLLAPSE:.+]] = tensor.collapse_shape %[[CONV_DISPATCH]]
// CHECK:         %[[BARRIER:.+]] = util.optimization_barrier %[[COLLAPSE]]
// CHECK:         %[[UNPACK_DISPATCH:.+]] = flow.dispatch.region
// CHECK:           %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSE]]
// CHECK:           flow.return %[[UNPACK]]
// CHECK:         util.return %[[UNPACK_DISPATCH]], %[[BARRIER]]
util.func public @no_fuse_multi_use_collapse(%input: tensor<1x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> (tensor<14x14x16xf32>, tensor<2x14x14x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x14x14x8xf32>) -> tensor<1x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<1x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<1x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<1x2x14x14x8xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3], [4]] : tensor<1x2x14x14x8xf32> into tensor<2x14x14x8xf32>
  %barrier = util.optimization_barrier %collapsed : tensor<2x14x14x8xf32>
  %dest = tensor.empty() : tensor<14x14x16xf32>
  %unpack = linalg.unpack %collapsed outer_dims_perm = [2, 0, 1] inner_dims_pos = [2] inner_tiles = [8] into %dest : tensor<2x14x14x8xf32> -> tensor<14x14x16xf32>
  util.return %unpack, %barrier : tensor<14x14x16xf32>, tensor<2x14x14x8xf32>
}

// -----

// A set_encoding joined the dispatch first, so the result unpack keeps its own
// dispatch.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
#id = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#encoding = #iree_encoding.testing<>
// AGGRESSIVE-LABEL: util.func public @no_fuse_unpack_after_set_encoding(
// AGGRESSIVE:         %[[DISPATCH:.+]]:2 = flow.dispatch.region
// AGGRESSIVE:           %[[CONV:.+]] = linalg.generic
// AGGRESSIVE:           %[[RELU:.+]] = linalg.generic
// AGGRESSIVE-SAME:        ins(%[[CONV]] :
// AGGRESSIVE:           %[[ENCODED:.+]] = iree_encoding.set_encoding %[[RELU]]
// AGGRESSIVE:           flow.return %[[RELU]], %[[ENCODED]]
// AGGRESSIVE:         %[[UNPACK_DISPATCH:.+]] = flow.dispatch.region
// AGGRESSIVE:           %[[UNPACK:.+]] = linalg.unpack %[[DISPATCH]]#0
// AGGRESSIVE:           flow.return %[[UNPACK]]
// AGGRESSIVE:         util.return %[[UNPACK_DISPATCH]], %[[DISPATCH]]#1
util.func public @no_fuse_unpack_after_set_encoding(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> (tensor<2x14x14x16xf32>, tensor<2x2x14x14x8xf32, #encoding>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %relu = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%conv : tensor<2x2x14x14x8xf32>) outs(%empty : tensor<2x2x14x14x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x2x14x14x8xf32>
  %dest = tensor.empty() : tensor<2x14x14x16xf32>
  %encoded = iree_encoding.set_encoding %relu : tensor<2x2x14x14x8xf32> -> tensor<2x2x14x14x8xf32, #encoding>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dest : tensor<2x2x14x14x8xf32> -> tensor<2x14x14x16xf32>
  util.return %unpack, %encoded : tensor<2x14x14x16xf32>, tensor<2x2x14x14x8xf32, #encoding>
}

// -----

// The result unpack joined the dispatch first, so the set_encoding stays
// outside of it.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
#id = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#encoding = #iree_encoding.testing<>
// AGGRESSIVE-LABEL: util.func public @no_fuse_set_encoding_after_unpack(
// AGGRESSIVE:         %[[DISPATCH:.+]]:2 = flow.dispatch.region
// AGGRESSIVE:           %[[CONV:.+]] = linalg.generic
// AGGRESSIVE:           %[[RELU:.+]] = linalg.generic
// AGGRESSIVE-SAME:        ins(%[[CONV]] :
// AGGRESSIVE:           %[[UNPACK:.+]] = linalg.unpack %[[RELU]]
// AGGRESSIVE:           flow.return %[[RELU]], %[[UNPACK]]
// AGGRESSIVE:         %[[ENCODED:.+]] = iree_encoding.set_encoding %[[DISPATCH]]#0
// AGGRESSIVE:         util.return %[[DISPATCH]]#1, %[[ENCODED]]
util.func public @no_fuse_set_encoding_after_unpack(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>) -> (tensor<2x14x14x16xf32>, tensor<2x2x14x14x8xf32, #encoding>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32>
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  %relu = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%conv : tensor<2x2x14x14x8xf32>) outs(%empty : tensor<2x2x14x14x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x2x14x14x8xf32>
  %dest = tensor.empty() : tensor<2x14x14x16xf32>
  %unpack = linalg.unpack %relu outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dest : tensor<2x2x14x14x8xf32> -> tensor<2x14x14x16xf32>
  %encoded = iree_encoding.set_encoding %relu : tensor<2x2x14x14x8xf32> -> tensor<2x2x14x14x8xf32, #encoding>
  util.return %unpack, %encoded : tensor<2x14x14x16xf32>, tensor<2x2x14x14x8xf32, #encoding>
}
