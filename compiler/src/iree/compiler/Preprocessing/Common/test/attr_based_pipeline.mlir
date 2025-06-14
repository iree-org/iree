// RUN: iree-opt --iree-preprocessing-attr-based-pipeline --mlir-print-local-scope --split-input-file --verify-diagnostics --iree-dispatch-creation-propagate-collapse-across-expands=true %s | FileCheck %s

func.func @single_dispatch_dropunitdims(%lhs : tensor<1x26x18x288xbf16>, %rhs :  tensor<288x288x3x3xbf16>, %outs : tensor<1x288x26x18xbf16>,
    %outs2 : tensor<1x288x24x16xf32>) -> tensor<1x288x24x16xf32> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">} {
  %transposed = linalg.transpose ins(%lhs : tensor<1x26x18x288xbf16>) outs(%outs : tensor<1x288x26x18xbf16>) permutation = [0, 3, 1, 2]
  %conv = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%transposed, %rhs : tensor<1x288x26x18xbf16>, tensor<288x288x3x3xbf16>)
    outs(%outs2 : tensor<1x288x24x16xf32>) -> tensor<1x288x24x16xf32>
  return %conv : tensor<1x288x24x16xf32>
}
// CHECK-LABEL: @single_dispatch_dropunitdims
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<1x26x18x288xbf16>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]]
//       CHECK:     %[[CONV:.+]] = linalg.generic {{.*}} ins(%[[COLLAPSE]]
//       CHECK:     flow.return %[[CONV]]
//       CHECK:   return %[[DISPATCH]]

// -----
func.func @single_dispatch_fusion(%lhs : tensor<18x288xf32>, %rhs :  tensor<18x288xbf16>, %outs : tensor<18x288xbf16>)
    -> tensor<18x288xbf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">} {
  %first = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%lhs : tensor<18x288xf32>) outs(%outs : tensor<18x288xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %1 = arith.truncf %in : f32 to bf16
    linalg.yield %1 : bf16
  } -> tensor<18x288xbf16>
  %final = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%first, %rhs : tensor<18x288xbf16>, tensor<18x288xbf16>) outs(%outs : tensor<18x288xbf16>) {
  ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
    %2 = arith.addf %in, %in_3 : bf16
    linalg.yield %2 : bf16
  } -> tensor<18x288xbf16>
  return %final : tensor<18x288xbf16>
}

// CHECK-LABEL: @single_dispatch_fusion
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic
//       CHECK:       %[[TRUNC:.+]] =  arith.truncf
//       CHECK:       %[[ADD:.+]] = arith.addf %[[TRUNC]]
//   CHECK-NOT:     linalg.generic

// -----
// Verifies the `tensor.expand_shape` op remains positioned after the elementwise op.

func.func @single_dispatch_bubble_and_sink_expand_shape(%arg0 : tensor<16x50x32x2048xbf16>, %arg1 : tensor<16x16x32x2048xbf16>, %arg2 : tensor<2048x3x2048xf32>)
    -> tensor<2048x3x1x2048xbf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">} {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4 * 3, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x50x32x2048xbf16>, tensor<16x16x32x2048xbf16>) outs(%arg2 : tensor<2048x3x2048xf32>) {
  ^bb0(%in: bf16, %in_1: bf16, %out: f32):
    %10 = arith.extf %in : bf16 to f32
    %11 = arith.extf %in_1 : bf16 to f32
    %12 = arith.mulf %10, %11 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<2048x3x2048xf32>
  %1 = tensor.empty() : tensor<2048x3x2048xbf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<2048x3x2048xf32>) outs(%1 : tensor<2048x3x2048xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %10 = arith.truncf %in : f32 to bf16
    linalg.yield %10 : bf16
  } -> tensor<2048x3x2048xbf16>
  %expanded = tensor.expand_shape %2 [[0], [1, 2], [3]] output_shape [2048, 3, 1, 2048] : tensor<2048x3x2048xbf16> into tensor<2048x3x1x2048xbf16>
  return %expanded : tensor<2048x3x1x2048xbf16>
}

// CHECK-LABEL: @single_dispatch_bubble_and_sink_expand_shape
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     linalg.generic
//   CHECK-NOT:     tensor.expand_shape
//       CHECK:     linalg.generic
//       CHECK:     tensor.expand_shape

// -----

module {
func.func @function1(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-generalize-linalg-matmul-experimental">} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  return %matmul : tensor<10x40xf16>
}
func.func @function2(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-pad-linalg-ops">} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  return %matmul : tensor<10x40xf16>
}
}
// CHECK-LABEL: func @function1
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//       CHECK:   return %[[GENERIC]]
// CHECK-LABEL: func @function2
//   CHECK-DAG:   %[[PAD1:.+]] = tensor.pad
//   CHECK-DAG:   %[[PAD2:.+]] = tensor.pad
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[PAD1]],
//  CHECK-SAME:       outs(%[[PAD2]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[MATMUL]]
//       CHECK:   return %[[SLICE]]

// -----

func.func @function(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf16>) -> tensor<10x40xf16> attributes {
    preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-pad-linalg-ops,iree-preprocessing-generalize-linalg-matmul-experimental">} {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf16>) -> tensor<10x40xf16>
  return %matmul : tensor<10x40xf16>
}
// CHECK-LABEL: func @function
//   CHECK-DAG:   %[[PAD1:.+]] = tensor.pad
//   CHECK-DAG:   %[[PAD2:.+]] = tensor.pad
//       CHECK:   %[[MATMUL:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[PAD1]],
//  CHECK-SAME:       outs(%[[PAD2]]
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[MATMUL]]
//       CHECK:   return %[[SLICE]]

// -----

// expected-remark@+1 {{expected preprocessing_pipeline attribute to be a `StringAttr` that specifies the pass pipeline to apply}}
func.func @function() attributes {
    preprocessing_pipeline = "iree-preprocessing-pad-linalg-ops"} {
  return
}
