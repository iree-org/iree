// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-dispatch-creation-pipeline{aggressive-fusion})" --mlir-print-local-scope %s | FileCheck %s

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
