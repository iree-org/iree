// RUN: iree-opt --iree-preprocessing-attr-based-pipeline --mlir-print-local-scope --split-input-file --verify-diagnostics %s | FileCheck %s

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
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]]
//       CHECK:     %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]]
//       CHECK:     %[[CONV:.+]] = linalg.generic {{.*}} ins(%[[EXPAND]]
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
