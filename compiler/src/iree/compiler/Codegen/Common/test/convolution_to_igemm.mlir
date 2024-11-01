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
// CHECK-DAG:    %[[IM2COL:.+]] = iree_linalg_ext.im2col {{.*}} : tensor<1x14x14x36xf32>) -> tensor<1x14x14x36xf32>
// CHECK-DAG:    %[[FILL:.+]] = linalg.fill {{.*}} -> tensor<1x14x14x16xf32>
// CHECK:        %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK:        %[[TRUNCF:.+]] = linalg.generic
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK:        return %[[TRUNCF]] : tensor<1x14x14x16xf16>

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
// CHECK-DAG:  %[[RES:.+]] = flow.dispatch.tensor.load {{.*}} -> tensor<1x14x14x16xf32>
// CHECK-DAG:  %[[IM2COL:.+]] = iree_linalg_ext.im2col {{.*}} ins(%[[LHS]] : tensor<1x16x16x4xf32>){{.*}}-> tensor<1x14x14x36xf32>
// CHECK-DAG:  %[[FILL:.+]] = linalg.fill {{.*}}outs(%[[RES]] : tensor<1x14x14x16xf32>)
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[RHS]] : tensor<1x14x14x36xf32>, tensor<36x16xf32>)
// CHECK-SAME:   outs(%[[FILL]] : tensor<1x14x14x16xf32>) {
// CHECK:      flow.dispatch.tensor.store %[[MATMUL]]

// -----

func.func @conv_with_lowering_config() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<3x3x128x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf32>> -> tensor<2x34x34x128xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x128x64xf32>> -> tensor<3x3x128x64xf32>
  %5 = tensor.empty() : tensor<2x32x32x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 8], subgroup = [1, 2, 2, 1, 0], workgroup = [1, 2, 2, 4, 0]}>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x34x34x128xf32>, tensor<3x3x128x64xf32>) outs(%6 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 64], strides = [1, 1, 1, 1] : tensor<2x32x32x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  return
}

// CHECK:      func.func @conv_with_lowering_config
// CHECK:        iree_linalg_ext.im2col
// CHECK:        linalg.generic
// CHECK-SAME:     lowering_config = {{.*}}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func public @no_conv_contraction(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %matmul = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<128x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.mulf %in, %in_0 : f32
    %1 = arith.addf %0, %out : f32
    linalg.yield %1 : f32
  } -> tensor<128x128xf32>
  return %matmul : tensor<128x128xf32>
}
// CHECK: func.func public @no_conv_contraction
// CHECK:   linalg.generic
