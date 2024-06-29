// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-pad-attention{pad-to-multiple-of=0,64,5,0,0}),cse)" %s | FileCheck %s --check-prefix=CHECK

// TODO: These tests should be moved to tiling.mlir when PartialReductionOpInterface is implemented for attention op.

func.func @static_attention(%query : tensor<48x1178x64xf16>, %key : tensor<48x1178x64xf16>, %value : tensor<48x1178x64xf16>) -> tensor<48x1178x64xf16> {
  %scale = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<48x1178x64xf16>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<48x1178x64xf16>, tensor<48x1178x64xf16>, tensor<48x1178x64xf16>, f16) outs(%0 : tensor<48x1178x64xf16>) -> tensor<48x1178x64xf16>
  return %1 : tensor<48x1178x64xf16>
}

// CHECK-LABEL: func.func @static_attention
// CHECK-SAME:   (%[[QUERY:.+]]: tensor<48x1178x64xf16>, %[[KEY:.+]]: tensor<48x1178x64xf16>, %[[VALUE:.+]]: tensor<48x1178x64xf16>)
// CHECK:        %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:        %[[PAD_Q:.+]] = tensor.pad %[[QUERY]] low[0, 0, 0] high[0, 38, 1]
// CHECK:        %[[PAD_K:.+]] = tensor.pad %[[KEY]] low[0, 0, 0] high[0, 0, 1]
// CHECK:        %[[ACC:.+]] = tensor.empty() : tensor<48x1216x64xf16>
// CHECK:        %[[ATTN:.+]] = iree_linalg_ext.attention ins(%[[PAD_Q]], %[[PAD_K]], %[[VALUE]], %[[ZERO]] : tensor<48x1216x65xf16>, tensor<48x1178x65xf16>, tensor<48x1178x64xf16>, f16)
// CHECK-SAME:                    outs(%[[ACC]] : tensor<48x1216x64xf16>) -> tensor<48x1216x64xf16>
// CHECK:        %[[RESULT:.+]] = tensor.extract_slice %[[ATTN]][0, 0, 0] [48, 1178, 64] [1, 1, 1]
// CHECK:        return %[[RESULT]]

// -----

func.func @dynamic_attention(%query : tensor<48x?x64xf16>, %key : tensor<48x1178x64xf16>, %value : tensor<48x1178x64xf16>) -> tensor<48x?x64xf16> {
  %scale = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %query, %c1 : tensor<48x?x64xf16>
  %0 = tensor.empty(%dim) : tensor<48x?x64xf16>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<48x?x64xf16>, tensor<48x1178x64xf16>, tensor<48x1178x64xf16>, f16) outs(%0 : tensor<48x?x64xf16>) -> tensor<48x?x64xf16>
  return %1 : tensor<48x?x64xf16>
}

// CHECK-DAG:  #[[MAP0:.+]] = affine_map<()[s0] -> (-s0 + (s0 ceildiv 64) * 64)>
// CHECK:      func.func @dynamic_attention
// CHECK-SAME:   (%[[QUERY:.+]]: tensor<48x?x64xf16>, %[[KEY:.+]]: tensor<48x1178x64xf16>, %[[VALUE:.+]]: tensor<48x1178x64xf16>)
// CHECK:        %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:        %[[DIM:.+]] = tensor.dim %[[QUERY]], %c1 : tensor<48x?x64xf16>
// CHECK:        %[[PAD_SIZE:.+]] = affine.apply #[[MAP0]]()[%[[DIM]]]
// CHECK:        %[[PAD_Q:.+]] = tensor.pad %[[QUERY]] low[0, 0, 0] high[0, %[[PAD_SIZE]], 1]
// CHECK:        %[[PAD_K:.+]] = tensor.pad %[[KEY]] low[0, 0, 0] high[0, 0, 1]
// CHECK:        %[[DIM_ACC:.+]] = tensor.dim %padded, %c1 : tensor<48x?x65xf16>
// CHECK:        %[[ACC:.+]] = tensor.empty(%[[DIM_ACC]]) : tensor<48x?x64xf16>
// CHECK:        %[[ATTN:.+]] = iree_linalg_ext.attention ins(%[[PAD_Q]], %[[PAD_K]], %[[VALUE]], %[[ZERO]] : tensor<48x?x65xf16>, tensor<48x1178x65xf16>, tensor<48x1178x64xf16>, f16)
// CHECK-SAME:                    outs(%[[ACC]] : tensor<48x?x64xf16>) -> tensor<48x?x64xf16>
