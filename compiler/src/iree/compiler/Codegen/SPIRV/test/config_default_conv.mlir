// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

// Convolution with consumer pointwise ops

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 112)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 32)>
#map3 = affine_map<(d0) -> (d0 * 2)>
#map4 = affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>
#map5 = affine_map<(d0)[s0] -> (-d0 + 32, s0)>
#map6 = affine_map<(d0)[s0] -> (-d0 + 112, s0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @nhwc_conv_pointwise_112x112x32 {
  hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export public @nhwc_conv_pointwise_112x112x32 layout(#pipeline_layout)
    builtin.module {
      func.func @nhwc_conv_pointwise_112x112x32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c112 = arith.constant 112 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
        %13 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>> -> tensor<1x112x112x32xf32>
        %14 = tensor.empty() : tensor<1x112x112x32xf32>
        %19 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
        %21 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 32], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>> -> tensor<3x3x3x32xf32>
        %24 = tensor.empty() : tensor<1x112x112x32xf32>
        %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        %26 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
            ins(%19, %21 : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%25 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        %27 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%26, %13 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%14 : tensor<1x112x112x32xf32>) {
            ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
              %28 = arith.subf %arg3, %arg4 : f32
              linalg.yield %28 : f32
            } -> tensor<1x112x112x32xf32>
        flow.dispatch.tensor.store %27, %3, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1]
            : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 32], [1, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @nhwc_conv_pointwise_112x112x32
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [8 : index, 2 : index, 2 : index]
//      CHECK: func.func @nhwc_conv_pointwise_112x112x32()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable private @nchw_conv_2x1280x8x8 {
  hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export public @nchw_conv_2x1280x8x8 layout(#pipeline_layout)
    builtin.module {
      func.func @nchw_conv_2x1280x8x8() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x1280x10x10xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1280x1280x3x3xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 1280, 10, 10], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280x10x10xf32>> -> tensor<2x1280x10x10xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1280, 1280, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280x3x3xf32>> -> tensor<1280x1280x3x3xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 1280, 8, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>> -> tensor<2x1280x8x8xf32>
        %6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x1280x10x10xf32>, tensor<1280x1280x3x3xf32>) outs(%5 : tensor<2x1280x8x8xf32>) -> tensor<2x1280x8x8xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [2, 1280, 8, 8], strides = [1, 1, 1, 1] : tensor<2x1280x8x8xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 16, 8, 8], [1, 8, 1, 4], [0, 0, 0, 0, 4, 1, 1], [0, 0, 1, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @nchw_conv_2x1280x8x8
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 8 : index, 2 : index]
//      CHECK: func.func @nchw_conv_2x1280x8x8()
//      CHECK:   linalg.conv_2d_nchw_fchw
// CHECK-SAME:     lowering_config = #[[CONFIG]]
