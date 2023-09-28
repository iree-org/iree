// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable private @nhwc_conv_pointwise_2x64x64x320 {
  hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>>
    }> {
    hal.executable.export public @nhwc_conv_pointwise_2x64x64x320 layout(#pipeline_layout)
    builtin.module {
      func.func @nhwc_conv_pointwise_2x64x64x320() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x66x66x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x64x64x320xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x64x64x320xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 66, 66, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x66x66x320xf16>> -> tensor<2x66x66x320xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 320, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>> -> tensor<3x3x320x320xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 64, 64, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64x64x320xf16>> -> tensor<2x64x64x320xf16>
        %7 = tensor.empty() : tensor<2x64x64x320xf16>
        %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16>
        %9 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
          ins(%4, %5 : tensor<2x66x66x320xf16>, tensor<3x3x320x320xf16>) outs(%8 : tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16>
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
          ins(%9, %6 : tensor<2x64x64x320xf16>, tensor<2x64x64x320xf16>) outs(%7 : tensor<2x64x64x320xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.divf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<2x64x64x320xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0, 0], sizes = [2, 64, 64, 320], strides = [1, 1, 1, 1] : tensor<2x64x64x320xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x64x64x320xf16>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 64, 64], [1, 1, 8, 8], [0, 0, 0, 0, 1, 1, 8], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @nhwc_conv_pointwise_2x64x64x320
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [8 : index, 8 : index, 1 : index]
//      CHECK: func.func @nhwc_conv_pointwise_2x64x64x320()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

