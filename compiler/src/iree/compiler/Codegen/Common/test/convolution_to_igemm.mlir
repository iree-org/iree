// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-convolution-to-igemm),canonicalize,cse)" %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3)->(d0, d1, d2, d3)>
func.func public @conv_with_consumer(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>) -> tensor<1x14x14x16xf16> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x14x14x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %1 = tensor.empty() : tensor<1x14x14x16xf16>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x14x14x16xf32>) outs(%1 : tensor<1x14x14x16xf16>) {
  ^bb0(%in: f32, %out: f16):
    %3 = arith.truncf %in : f32 to f16
    linalg.yield %3 : f16
  } -> tensor<1x14x14x16xf16>
  return %2 : tensor<1x14x14x16xf16>
}
// CHECK:      func.func public @conv_with_consumer
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   : tensor<1x196x36xf32>) -> tensor<1x196x36xf32>
// CHECK:      %[[FILL:.+]] = linalg.fill
// CHECK-SAME:   -> tensor<1x196x16xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK:      %[[TRUNCF:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
// CHECK:      %[[EXPANDED:.+]] = tensor.expand_shape %[[TRUNCF]] {{\[}}[0], [1, 2], [3]] output_shape [1, 14, 14, 16] : tensor<1x196x16xf16> into tensor<1x14x14x16xf16>
// CHECK:      return %[[EXPANDED]] : tensor<1x14x14x16xf16>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_gpu.lowering_config<{thread = [2, 16], subgroup = [2, 16]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fold_with_interface_tensor() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x16x16x4xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x4x16xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 16, 16, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x16x16x4xf32>> -> tensor<1x16x16x4xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x4x16xf32>> -> tensor<3x3x4x16xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>> -> tensor<1x14x14x16xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    %6 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%3, %4: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 16], strides = [1, 1, 1, 1] : tensor<1x14x14x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>>
    return
  }
}

// CHECK:      func.func @fold_with_interface_tensor
// CHECK-DAG:  %[[LHS:.+]] = flow.dispatch.tensor.load {{.*}} -> tensor<1x16x16x4xf32>
// CHECK-DAG:  %[[RHS:.+]] = flow.dispatch.tensor.load {{.*}} -> tensor<36x16xf32>
// CHECK-DAG:  %[[RES:.+]] = flow.dispatch.tensor.load {{.*}} -> tensor<1x196x16xf32>
// CHECK-DAG:  %[[IM2COL:.+]] = iree_linalg_ext.im2col {{.*}} ins(%[[LHS]] : tensor<1x16x16x4xf32>){{.*}}-> tensor<1x196x36xf32>
// CHECK-DAG:  %[[FILL:.+]] = linalg.fill {{.*}}outs(%[[RES]] : tensor<1x196x16xf32>)
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[RHS]] : tensor<1x196x36xf32>, tensor<36x16xf32>)
// CHECK-SAME:   outs(%[[FILL]] : tensor<1x196x16xf32>) {
// CHECK:      flow.dispatch.tensor.store %[[MATMUL]]

// -----

#map = affine_map<(d0, d1, d2, d3)->(d0, d1, d2, d3)>
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
func.func public @conv_with_lowering_config(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>) -> tensor<1x14x14x16xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x14x14x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %0 = linalg.conv_2d_nhwc_hwcf {lowering_config = #config,
      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  return %0 : tensor<1x14x14x16xf32>
}
// CHECK:      func.func public @conv_with_lowering_config
// CHECK-NOT:    iree_linalg_ext.im2col
// CHECK:        linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config
