// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-flow-dispatch-linalg-on-tensors-pass), cse, canonicalize, cse)" %s | FileCheck %s

func.func @no_fuse_quantized(%arg0 : tensor<?x113x113x64xi8>, %arg1 : tensor<3x3x64xi8>,
    %arg2 : i32, %arg3 : i32) -> tensor<?x56x56x64xi8> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x113x113x64xi8>
  %0 = tensor.empty(%d0) : tensor<?x56x56x64xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x56x56x64xi32>) -> tensor<?x56x56x64xi32>
  %2 =  linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x113x113x64xi8>, tensor<3x3x64xi8>, i32, i32)
      outs(%1 : tensor<?x56x56x64xi32>) -> tensor<?x56x56x64xi32>
  %3 = tensor.empty(%d0) : tensor<?x56x56x64xi8>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%2 : tensor<?x56x56x64xi32>) outs(%3 : tensor<?x56x56x64xi8>) {
    ^bb0(%b0: i32, %b1 : i8):
      %5 = arith.trunci %b0 : i32 to i8
      linalg.yield %5 : i8
    } -> tensor<?x56x56x64xi8>
  return %4 : tensor<?x56x56x64xi8>
}
//     CHECK: func.func @no_fuse_quantized
//     CHECK:   flow.dispatch.workgroups
//     CHECK:   linalg.depthwise_conv_2d_nhwc_hwc_q
// CHECK-NOT:   linalg.generic
//     CHECK:   flow.dispatch.workgroups
//     CHECK:   linalg.generic

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_broadcast_elementwise_type_mismatch(%a: tensor<12x16x16xf32>, %b: tensor<12x16x32xf32>) -> tensor<12x16x32xi32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = tensor.empty() : tensor<12x16xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<12x16xf32>) -> tensor<12x16xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<12x16x16xf32>) outs(%38 : tensor<12x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %780 = arith.maxf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16xf32>
  %40 = tensor.empty() : tensor<12x16x32xi32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%b, %39 : tensor<12x16x32xf32>, tensor<12x16xf32>) outs(%40 : tensor<12x16x32xi32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: i32):
    %780 = arith.subf %arg3, %arg4 : f32
    %781 = arith.fptosi %780 : f32 to i32
    linalg.yield %781 : i32
  } -> tensor<12x16x32xi32>
  return %42 : tensor<12x16x32xi32>
}

// Check that two generic ops are NOT dispatched together since the input type
// for reduction is different from the output type of the elementwise op. We
// should see two flow.dispatch.workgroups.

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_type_mismatch
//      CHECK: flow.dispatch.workgroups
//      CHECK: flow.dispatch.workgroups

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_set_encoding(%arg0: tensor<512xf32>, %arg1: tensor<384x512xf32>,
    %arg2: tensor<384x512xf32>) -> tensor<384x512xf32, #iree_linalg_ext.encoding<GEMM_LHS>> {
  %0 = tensor.empty() : tensor<384x512xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1],
                       iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1, %arg2 : tensor<512xf32>, tensor<384x512xf32>, tensor<384x512xf32>)
    outs(%0 : tensor<384x512xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_0 : f32
    %4 = arith.addf %3, %in_1 : f32
    linalg.yield %4 : f32
  } -> tensor<384x512xf32>
  %2 = iree_linalg_ext.set_encoding %1 : tensor<384x512xf32> -> tensor<384x512xf32, #iree_linalg_ext.encoding<GEMM_LHS>>
  return %2 : tensor<384x512xf32, #iree_linalg_ext.encoding<GEMM_LHS>>
}
// CHECK-LABEL: func.func @elem_set_encoding
// CHECK:         flow.dispatch.workgroups
// CHECK:           linalg.generic
// CHECK:           iree_linalg_ext.set_encoding
// CHECK-NOT:     flow.dispatch.workgroups
