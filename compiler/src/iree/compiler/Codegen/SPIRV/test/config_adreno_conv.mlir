// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

// Conv - large OC - distribute to only one workgroup dimension.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @conv_112x112x512 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export public @conv_112x112x512 layout(#pipeline_layout)
    builtin.module {
      func.func @conv_112x112x512() {
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c112 = arith.constant 112 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x3x512xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x512xf32>>
        %13 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
        %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 512], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x3x512xf32>> -> tensor<3x3x3x512xf32>
        %22 = tensor.empty() : tensor<1x112x112x512xf32>
        %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
        %24 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
            ins(%13, %15 : tensor<1x225x225x3xf32>, tensor<3x3x3x512xf32>) outs(%23 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
        flow.dispatch.tensor.store %24, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 512], strides = [1, 1, 1, 1]
            : tensor<1x112x112x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x512xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 8, 256], [1, 1, 8, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @conv_112x112x512
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 1 : index, 1 : index]
//      CHECK: func.func @conv_112x112x512()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Conv - medium OC/OW/OH - distribute to two workgroup dimensions.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @conv_112x112x32 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export public @conv_112x112x32 layout(#pipeline_layout)
    builtin.module {
      func.func @conv_112x112x32() {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c112 = arith.constant 112 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
        %13 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
        %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 32], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>> -> tensor<3x3x3x32xf32>
        %22 = tensor.empty() : tensor<1x112x112x32xf32>
        %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        %24 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
            ins(%13, %15 : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%23 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        flow.dispatch.tensor.store %24, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1]
            : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 16, 32], [1, 4, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @conv_112x112x32
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [8 : index, 8 : index, 1 : index]
//      CHECK: func.func @conv_112x112x32()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Conv - small OC/OW/OH - distribute to all three workgroup dimensions.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @conv_16x16x16 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export public @conv_16x16x16 layout(#pipeline_layout)
    builtin.module {
      func.func @conv_16x16x16() {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x33x33x3xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x16x16x16xf32>>
        %13 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 33, 33, 3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x33x33x3xf32>> -> tensor<1x33x33x3xf32>
        %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
        %22 = tensor.empty() : tensor<1x16x16x16xf32>
        %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
        %24 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
            ins(%13, %15 : tensor<1x33x33x3xf32>, tensor<3x3x3x16xf32>) outs(%23 : tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
        flow.dispatch.tensor.store %24, %2, offsets = [0, 0, 0, 0], sizes = [1, 16, 16, 16], strides = [1, 1, 1, 1]
            : tensor<1x16x16x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x16x16x16xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 8, 8, 16], [1, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @conv_16x16x16
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [4 : index, 4 : index, 4 : index]

//      CHECK: func.func @conv_16x16x16()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Depthwise conv - small OC/OW/OH - distribute to all three workgroup dimensions.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dwconv_28x28x144 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export public @dwconv_28x28x144 layout(#pipeline_layout)
    builtin.module {
      func.func @dwconv_28x28x144() {
        %c0 = arith.constant 0 : index
        %c144 = arith.constant 144 : index
        %c28 = arith.constant 28 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x57x57x144xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x144xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x28x28x144xf32>>
        %14 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 57, 57, 144], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x57x57x144xf32>> -> tensor<1x57x57x144xf32>
        %16 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 144], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x144xf32>> -> tensor<3x3x144xf32>
        %23 = tensor.empty() : tensor<1x28x28x144xf32>
        %24 = linalg.fill ins(%cst : f32) outs(%23 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
        %25 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
                  ins(%14, %16 : tensor<1x57x57x144xf32>, tensor<3x3x144xf32>) outs(%24 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
        flow.dispatch.tensor.store %25, %2, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 144], strides = [1, 1, 1, 1]
            : tensor<1x28x28x144xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x28x28x144xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 16], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @dwconv_28x28x144
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [4 : index, 4 : index, 4 : index]
//      CHECK: func.func @dwconv_28x28x144()
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Depthwise conv - tiny OC/OW/OH - starving the GPU.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dwconv_4x4x8 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export public @dwconv_4x4x8 layout(#pipeline_layout)
    builtin.module {
      func.func @dwconv_4x4x8() {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x9x9x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x8xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x4x4x8xf32>>
        %14 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x9x9x8xf32>> -> tensor<1x9x9x8xf32>
        %16 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 8], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x8xf32>> -> tensor<3x3x8xf32>
        %23 = tensor.empty() : tensor<1x4x4x8xf32>
        %24 = linalg.fill ins(%cst : f32) outs(%23 : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
        %25 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
            ins(%14, %16 : tensor<1x9x9x8xf32>, tensor<3x3x8xf32>) outs(%24 : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
        flow.dispatch.tensor.store %25, %2, offsets = [0, 0, 0, 0], sizes = [1, 4, 4, 8], strides = [1, 1, 1, 1]
            : tensor<1x4x4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x4x4x8xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 8], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @dwconv_4x4x8
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 4 : index, 4 : index]
//      CHECK: func.func @dwconv_4x4x8()
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]
