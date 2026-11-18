// RUN: iree-opt --split-input-file --iree-dispatch-creation-transpose-generic-ops -canonicalize -cse --mlir-print-local-scope %s | FileCheck %s

util.func @supported_conv(%arg0 : tensor<2x130x130x16xf16>, %arg1 : tensor<3x3x16x320xf16>) -> tensor<2x320x128x128xf16> {
  %empty = tensor.empty() : tensor<2x128x128x320xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%arg0, %arg1 : tensor<2x130x130x16xf16>, tensor<3x3x16x320xf16>)
      outs(%fill : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  %empty1 = tensor.empty() : tensor<2x320x128x128xf16>
  %truncf = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv : tensor<2x128x128x320xf32>) outs(%empty1 : tensor<2x320x128x128xf16>) {
    ^bb0(%b0 : f32, %b1 :f16):
      %0 = arith.truncf %b0 : f32 to f16
      linalg.yield %0 : f16
  } -> tensor<2x320x128x128xf16>
  util.return %truncf : tensor<2x320x128x128xf16>
}
// CHECK-LABEL: func public @supported_conv(
//       CHECK:   %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>]
//  CHECK-SAME:       ins(%[[CONV]] :
//       CHECK:   return %[[GENERIC]]

// -----

util.func @generalize_to_any_linalg_op(%arg0 : tensor<?x?x?x?xi8>, %arg1 : tensor<?x?x?x?xi8>,
    %arg2 : tensor<?x?x?x?xi64>, %arg3 : tensor<?x?x?x?xi64>, %arg4 : tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xi8> {
  %c0_i64 = arith.constant 0 : i64
  %0 = linalg.conv_2d_nhwc_hwcf_q {
      dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%arg0, %arg1, %c0_i64, %c0_i64 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, i64, i64)
      outs(%arg2 : tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0 : tensor<?x?x?x?xi64>) outs(%arg4 : tensor<?x?x?x?xi8>) {
  ^bb0(%in: i64, %out: i8):
    %3 = arith.trunci %in : i64 to i32
    %4 = arith.sitofp %3 : i32 to f32
    %5 = arith.fptosi %4 : f32 to i8
    linalg.yield %5 : i8
  } -> tensor<?x?x?x?xi8>
  util.return %2 : tensor<?x?x?x?xi8>
}
// CHECK-LABEL: func public @generalize_to_any_linalg_op(
//       CHECK:   %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf_q
//       CHECK:   %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>]
//       CHECK:   return %[[RESULT]]

//  -----

//      CHECK: util.func public @interchange
//      CHECK:   linalg.generic {indexing_maps = [
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d3, d0, d1)>
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
util.func public @interchange(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>],
    iterator_types = ["reduction", "parallel", "parallel", "parallel"]}
  ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  outs(%arg2 : tensor<?x?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %m = arith.mulf %arg3, %arg4 : f32
    %a = arith.addf %arg5, %m : f32
    linalg.yield %a : f32
  } -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}

//  -----

//      CHECK: util.func public @swap_two_reduction_dims
//      CHECK:   linalg.generic {indexing_maps = [
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d2, d5)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
util.func public @swap_two_reduction_dims(%arg0: tensor<16x2x48x32x288xbf16>, %arg1: tensor<288x2x288xbf16>, %arg2: tensor<16x48x32x288xf32>) -> tensor<16x48x32x288xf32> {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d2, d4)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d5, d4)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
  ins(%arg0, %arg1 : tensor<16x2x48x32x288xbf16>, tensor<288x2x288xbf16>)
  outs(%arg2 : tensor<16x48x32x288xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %1 = arith.extf %in : bf16 to f32
    %2 = arith.extf %in_0 : bf16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x48x32x288xf32>
  util.return %0 : tensor<16x48x32x288xf32>
}

//  -----

//      CHECK: util.func public @already_sorted_reductions
//      CHECK:   linalg.generic {indexing_maps = [
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d2, d5)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
util.func public @already_sorted_reductions(%arg0: tensor<16x2x48x32x288xbf16>, %arg1: tensor<288x2x288xbf16>, %arg2: tensor<16x48x32x288xf32>) -> tensor<16x48x32x288xf32> {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d2, d5)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
  ins(%arg0, %arg1 : tensor<16x2x48x32x288xbf16>, tensor<288x2x288xbf16>)
  outs(%arg2 : tensor<16x48x32x288xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %1 = arith.extf %in : bf16 to f32
    %2 = arith.extf %in_0 : bf16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x48x32x288xf32>
  util.return %0 : tensor<16x48x32x288xf32>
}

//  -----

//      CHECK: util.func public @single_reduction_dim
//      CHECK:   linalg.generic {indexing_maps = [
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
util.func public @single_reduction_dim(%arg0: tensor<4x8x16xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<4x16x32xf32>) -> tensor<4x16x32xf32> {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<32x8xf32>)
  outs(%arg2 : tensor<4x16x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<4x16x32xf32>
  util.return %0 : tensor<4x16x32xf32>
}

//  -----

//      CHECK: util.func public @three_reduction_dims_swap_middle
//      CHECK:   linalg.generic {indexing_maps = [
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4, d5)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
util.func public @three_reduction_dims_swap_middle(%arg0: tensor<4x8x16x32xf32>, %arg1: tensor<64x8x16x32xf32>, %arg2: tensor<4x64x128xf32>) -> tensor<4x64x128xf32> {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d4)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5, d4)>,
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
  ins(%arg0, %arg1 : tensor<4x8x16x32xf32>, tensor<64x8x16x32xf32>)
  outs(%arg2 : tensor<4x64x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<4x64x128xf32>
  util.return %0 : tensor<4x64x128xf32>
}

//  -----

//      CHECK: util.func public @non_contiguous_reductions_swap
//      CHECK:   linalg.generic {indexing_maps = [
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>,
// CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]}
util.func public @non_contiguous_reductions_swap(%arg0: tensor<8x16x32x64xf32>, %arg1: tensor<128x16x32x64xf32>, %arg2: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>,
    affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d4, d3)>,
    affine_map<(d0, d1, d2, d3, d4) -> (d0, d2)>],
    iterator_types = ["parallel", "reduction", "parallel", "reduction", "reduction"]}
  ins(%arg0, %arg1 : tensor<8x16x32x64xf32>, tensor<128x16x32x64xf32>)
  outs(%arg2 : tensor<8x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<8x128xf32>
  util.return %0 : tensor<8x128xf32>
}

// -----

// Without materialized layouts, the reduction loops of a data-tiled
// convolution move innermost like for any other generic.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
// CHECK-LABEL: util.func public @data_tiled_conv_interchange(
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
util.func public @data_tiled_conv_interchange(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>, %init: tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32> {
  %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
    ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %mul, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<2x2x14x14x8xf32>
  util.return %conv : tensor<2x2x14x14x8xf32>
}

// -----

// Data-tiled convolutions from materialized layouts keep their loop order.
#input = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, ic, h + kh, w + kw, i)>
#filter = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (oc, ic, kh, kw, i, o)>
#output = affine_map<(n, oc, h, w, ic, kh, kw, o, i) -> (n, oc, h, w, o)>
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {iree.encoding.materialized_layout_target = #target} {
  // CHECK-LABEL: util.func public @no_interchange_materialized_data_tiled_conv(
  // CHECK:         linalg.generic
  // CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]
  util.func public @no_interchange_materialized_data_tiled_conv(%input: tensor<2x1x16x16x8xf32>, %filter: tensor<2x1x3x3x8x8xf32>, %init: tensor<2x2x14x14x8xf32>) -> tensor<2x2x14x14x8xf32> {
    %conv = linalg.generic {indexing_maps = [#input, #filter, #output],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
      ins(%input, %filter : tensor<2x1x16x16x8xf32>, tensor<2x1x3x3x8x8xf32>) outs(%init : tensor<2x2x14x14x8xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %mul = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %mul, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<2x2x14x14x8xf32>
    util.return %conv : tensor<2x2x14x14x8xf32>
  }
}
