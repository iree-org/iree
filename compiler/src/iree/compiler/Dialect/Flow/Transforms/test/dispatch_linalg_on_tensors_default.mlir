// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions, iree-flow-clone-producers-into-dispatch-regions, iree-flow-form-dispatch-workgroups), cse, canonicalize, cse)" %s | FileCheck %s

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

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_set_encoding(%arg0: tensor<512xf32>, %arg1: tensor<384x512xf32>,
    %arg2: tensor<384x512xf32>) -> tensor<384x512xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>> {
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
  %2 = iree_linalg_ext.set_encoding %1 : tensor<384x512xf32> -> tensor<384x512xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
  return %2 : tensor<384x512xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
}
// CHECK-LABEL: func.func @elem_set_encoding
// CHECK:         flow.dispatch.workgroups
// CHECK:           linalg.generic
// CHECK:           iree_linalg_ext.set_encoding
// CHECK-NOT:     flow.dispatch.workgroups

// -----

func.func @fix_dominance_on_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %matmul = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %bias_add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul, %arg2 : tensor<?x?xf32>, tensor<?xf32>) outs(%empty : tensor<?x?xf32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 :f32):
      %addf = arith.addf %b0, %b1 : f32
      linalg.yield %addf : f32
  } -> tensor<?x?xf32>
  return %bias_add : tensor<?x?xf32>
}
// CHECK-LABEL: func @fix_dominance_on_fusion
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[EMPTY]] :
//       CHECK:     %[[GEMM:.+]] = linalg.matmul
//  CHECK-SAME:         outs(%[[FILL]] :
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GEMM]],
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC]]
//       CHECK:   return %[[RESULT]]
