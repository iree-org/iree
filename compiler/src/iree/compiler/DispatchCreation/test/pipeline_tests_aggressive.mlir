// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-dispatch-creation-pipeline{aggressive-fusion})" --mlir-print-local-scope %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-dispatch-creation-pipeline{aggressive-fusion aggressive-reshape-movement})" --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK-AGGRESSIVE-RESHAPE

util.func public @truncate_fusion(%arg0: tensor<2x64x64x320xi8>, %arg1: tensor<2x66x66x640xi8>, %arg2: tensor<3x3x640x640xi8>, %arg3: tensor<640xi32>, %arg4: tensor<640xf32>, %arg5: tensor<640x320xi8>, %arg6: tensor<640xi32>, %arg7: tensor<640xf32>) -> tensor<2x640x64x64xf16> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<2x64x64x320xi8>
  %1 = tensor.empty() : tensor<2x64x64x640xi32>
  %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<2x64x64x640xi32>) -> tensor<2x64x64x640xi32>
  %3 = tensor.empty() : tensor<2x64x64x640xf32>
  %4 = tensor.empty() : tensor<2x640x64x64xf16>
  %5 = tensor.empty() : tensor<2x64x64x640xf16>
  %6 = tensor.empty() : tensor<2x64x64x320xf16>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x66x66x640xi8>, tensor<3x3x640x640xi8>) outs(%2 : tensor<2x64x64x640xi32>) -> tensor<2x64x64x640xi32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %arg3 : tensor<2x64x64x640xi32>, tensor<640xi32>) outs(%1 : tensor<2x64x64x640xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %19 = arith.addi %in, %in_0 : i32
    linalg.yield %19 : i32
  } -> tensor<2x64x64x640xi32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<2x64x64x640xi32>) outs(%3 : tensor<2x64x64x640xf32>) {
  ^bb0(%in: i32, %out: f32):
    %19 = arith.sitofp %in : i32 to f32
    linalg.yield %19 : f32
  } -> tensor<2x64x64x640xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %arg4 : tensor<2x64x64x640xf32>, tensor<640xf32>) outs(%3 : tensor<2x64x64x640xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.mulf %in, %in_0 : f32
    linalg.yield %19 : f32
  } -> tensor<2x64x64x640xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<2x64x64x640xf32>) outs(%5 : tensor<2x64x64x640xf16>) {
  ^bb0(%in: f32, %out: f16):
    %19 = arith.truncf %in : f32 to f16
    linalg.yield %19 : f16
  } -> tensor<2x64x64x640xf16>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%arg0, %arg5 : tensor<2x64x64x320xi8>, tensor<640x320xi8>) outs(%2 : tensor<2x64x64x640xi32>) {    ^bb0(%in: i8, %in_0: i8, %out: i32):
    %19 = arith.extsi %in : i8 to i32
    %20 = arith.extsi %in_0 : i8 to i32
    %21 = arith.muli %19, %20 : i32
    %22 = arith.addi %out, %21 : i32
    linalg.yield %22 : i32
  } -> tensor<2x64x64x640xi32>
  %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %arg6 : tensor<2x64x64x640xi32>, tensor<640xi32>) outs(%1 : tensor<2x64x64x640xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %19 = arith.addi %in, %in_0 : i32
    linalg.yield %19 : i32
  } -> tensor<2x64x64x640xi32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<2x64x64x640xi32>) outs(%3 : tensor<2x64x64x640xf32>) {
  ^bb0(%in: i32, %out: f32):
    %19 = arith.sitofp %in : i32 to f32
    linalg.yield %19 : f32
  } -> tensor<2x64x64x640xf32>
  %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %arg7 : tensor<2x64x64x640xf32>, tensor<640xf32>) outs(%3 : tensor<2x64x64x640xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.mulf %in, %in_0 : f32
    linalg.yield %19 : f32
  } -> tensor<2x64x64x640xf32>
  %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<2x64x64x640xf32>) outs(%5 : tensor<2x64x64x640xf16>) {
  ^bb0(%in: f32, %out: f16):
    %19 = arith.truncf %in : f32 to f16
    linalg.yield %19 : f16
  } -> tensor<2x64x64x640xf16>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %11 : tensor<2x64x64x640xf16>, tensor<2x64x64x640xf16>) outs(%5 : tensor<2x64x64x640xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %19 = arith.addf %in, %in_0 : f16
    linalg.yield %19 : f16
  } -> tensor<2x64x64x640xf16>
  %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<2x64x64x640xf16>) outs(%4 : tensor<2x640x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<2x640x64x64xf16>
  util.return %18 : tensor<2x640x64x64xf16>
}

// CHECK-LABEL: func public @truncate_fusion
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[MUL:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       outs(%{{.*}} : tensor<8192x640xi32>)
//       CHECK:     %[[TRUNC0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:       ins(%[[MUL]]
//  CHECK-SAME:       outs(%{{.*}} : tensor<8192x640xf16>)
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[TRUNC0]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {{.*}} -> tensor<2x64x64x640xi32>
//       CHECK:     %[[TRUNC1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%{{[a-zA-Z0-9]+}}, %[[CONV]]
//  CHECK-SAME:       outs(%{{.*}} : tensor<2x640x64x64xf16>)
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[TRUNC1]]

// -----

util.func public @attention_broadcast(
  %arg0 : tensor<32x16x?x128xf16>,
  %arg1 : tensor<4x?x8x128xf8E4M3FN>,
  %arg2 : tensor<4x?x8x128xf8E4M3FN>,
  %arg4 : tensor<32x16x?x?xf16>) -> (tensor<32x16x?x128xf16>) {
  %cst = arith.constant 1 : index
  %dim = tensor.dim %arg1, %cst : tensor<4x?x8x128xf8E4M3FN>
  %empty1 = tensor.empty(%dim) : tensor<4x?x8x128xf16>
  %empty2 = tensor.empty(%dim) : tensor<4x8x16x?x128xf16>
  %empty3 = tensor.empty(%dim) : tensor<4x8x16x128x?xf16>
  %empty4 = tensor.empty(%dim) : tensor<32x16x?x128xf16>
  %k = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg1 : tensor<4x?x8x128xf8E4M3FN>)
    outs(%empty1 : tensor<4x?x8x128xf16>){
  ^bb0(%in: f8E4M3FN, %out: f16):
    %extf = arith.extf %in : f8E4M3FN to f16
    linalg.yield %extf : f16
  } -> tensor<4x?x8x128xf16>
  %k_bcast = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%k : tensor<4x?x8x128xf16>)
    outs(%empty2 : tensor<4x8x16x?x128xf16>){
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x8x16x?x128xf16>
  %k_collapse = tensor.collapse_shape %k_bcast [[0, 1], [2], [3], [4]] : tensor<4x8x16x?x128xf16> into tensor<32x16x?x128xf16>
  %v = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg2 : tensor<4x?x8x128xf8E4M3FN>)
    outs(%empty1 : tensor<4x?x8x128xf16>){
  ^bb0(%in: f8E4M3FN, %out: f16):
    %extf = arith.extf %in : f8E4M3FN to f16
    linalg.yield %extf : f16
  } -> tensor<4x?x8x128xf16>
  %v_bcast = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%v : tensor<4x?x8x128xf16>)
    outs(%empty3 : tensor<4x8x16x128x?xf16>){
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x8x16x128x?xf16>
  %v_collapse = tensor.collapse_shape %v_bcast [[0, 1], [2], [3], [4]] : tensor<4x8x16x128x?xf16> into tensor<32x16x128x?xf16>
  %17 = iree_linalg_ext.attention {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]}
    ins(%arg0, %k_collapse, %v_collapse, %arg4 : tensor<32x16x?x128xf16>, tensor<32x16x?x128xf16>, tensor<32x16x128x?xf16>, tensor<32x16x?x?xf16>)
    outs(%empty4 : tensor<32x16x?x128xf16>) {
  ^bb0(%arg8: f32):
      iree_linalg_ext.yield %arg8 : f32
  } -> tensor<32x16x?x128xf16>
  util.return %17 : tensor<32x16x?x128xf16>
}
// Make sure that the broadcast gets fused with the attention op and not the
// producer linalg.generic ops.

// CHECK-LABEL: func public @attention_broadcast
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[Q:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
//       CHECK:     %[[K:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
//       CHECK:     iree_linalg_ext.attention
//  CHECK-SAME:       ins({{.*}}, %[[Q]], %[[K]], {{.*}} : tensor<4x8x?x128xf16>, tensor<4x?x8x128xf16>, tensor<4x?x8x128xf16>, tensor<4x8x?x?xf16>)

// -----

util.func public @transpose_barrier_matmul(%arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %empty_transpose = tensor.empty() : tensor<64x128xf32>
  %transpose = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<128x64xf32>) outs(%empty_transpose : tensor<64x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<64x128xf32>
  %barrier = iree_tensor_ext.compute_barrier.start %transpose : tensor<64x128xf32> -> tensor<64x128xf32>
  %empty_matmul = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%c0 : f32) outs(%empty_matmul : tensor<128x128xf32>) -> tensor<128x128xf32>
  %matmul = linalg.matmul ins(%arg1, %barrier : tensor<128x64xf32>, tensor<64x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  util.return %matmul : tensor<128x128xf32>
}
// CHECK-LABEL: func public @transpose_barrier_matmul
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[TRANSPOSE:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[TRANSPOSE]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.workgroups
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:       ins({{.*}}, {{.*}} : tensor<128x64xf32>, tensor<64x128xf32>)

// -----

// Test aggressive reshape movement: collapse shape should move through reduction so that the transpose and reduction can be fused.
util.func public @aggressive_reshape_movement(%arg0: tensor<2x10x20xi64>) -> tensor<10xi64> {
  %empty_transpose = tensor.empty() : tensor<2x20x10xi64>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x10x20xi64>) outs(%empty_transpose : tensor<2x20x10xi64>) permutation = [0, 2, 1]
  %collapsed = tensor.collapse_shape %transposed [[0, 1], [2]] : tensor<2x20x10xi64> into tensor<40x10xi64>
  %empty_reduction = tensor.empty() : tensor<10xi64>
  %reduced = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%collapsed : tensor<40x10xi64>) outs(%empty_reduction : tensor<10xi64>) {
  ^bb0(%in: i64, %out: i64):
    %x = arith.addi %in, %out : i64
    linalg.yield %x : i64
  } -> tensor<10xi64>
  util.return %reduced : tensor<10xi64>
}
// CHECK-LABEL: func public @aggressive_reshape_movement
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]:
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.workgroups(%[[ARG0]])
//       CHECK:   %[[RESHAPED:.+]] = flow.tensor.reshape %[[DISPATCH0]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.workgroups(%[[RESHAPED]])
//       CHECK:   util.return %[[DISPATCH1]]

// CHECK-AGGRESSIVE-RESHAPE-LABEL: util.func public @aggressive_reshape_movement
// CHECK-AGGRESSIVE-RESHAPE-SAME:    %[[ARG0:[a-zA-Z0-9]+]]:
// CHECK-AGGRESSIVE-RESHAPE:   %[[DISPATCH:.+]] = flow.dispatch.workgroups(%[[ARG0]])
// CHECK-AGGRESSIVE-RESHAPE:   util.return %[[DISPATCH]]
