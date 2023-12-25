// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-trim-executable-target-env)))' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8, GroupNonUniformArithmetic],
                                      [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class]>,
                                      api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<>>}>


// CHECK-DAG: #[[$TARGET0:.+]] = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>}>
// CHECK-DAG: #[[$TARGET1:.+]] = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniformArithmetic], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>}>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>

hal.executable private @predict_dispatch_0 {
  // CHECK-LABEL: hal.executable.variant public @vulkan_spirv_fb0
  //  CHECK-SAME: target(#[[$TARGET0]])
  hal.executable.variant public @vulkan_spirv_fb0 target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export public @predict_dispatch_0_vecmat_128x784_f32 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      hal.return %c2, %c1, %c1 : index, index, index
    }
    // CHECK-NOT: spirv.target_env
    builtin.module attributes {spirv.target_env = #spirv.target_env<
        #spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8],
        [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class]>,
        api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<>>} {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spirv.func @predict_dispatch_0_vecmat_128x784_f32() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @predict_dispatch_0_vecmat_128x784_f32
        spirv.ExecutionMode @predict_dispatch_0_vecmat_128x784_f32 "LocalSize", 64, 1, 1
      }
    }
  }
}

hal.executable private @predict_dispatch_1 {
  // CHECK-LABEL: hal.executable.variant public @vulkan_spirv_fb1
  //  CHECK-SAME: target(#[[$TARGET1]])
  hal.executable.variant public @vulkan_spirv_fb1 target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export public @predict_dispatch_1_vecmat_10x128_f32 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      hal.return %c10, %c1, %c1 : index, index, index
    }
    // CHECK-NOT: spirv.target_env
    builtin.module attributes {spirv.target_env = #spirv.target_env<
        #spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8],
        [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class]>,
        api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<>>} {
      spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader, GroupNonUniformArithmetic], [SPV_KHR_storage_buffer_storage_class]> {
        spirv.func @predict_dispatch_1_vecmat_10x128_f32() "None" {
          spirv.Return
        }
        spirv.EntryPoint "GLCompute" @predict_dispatch_1_vecmat_10x128_f32
        spirv.ExecutionMode @predict_dispatch_1_vecmat_10x128_f32 "LocalSize", 64, 1, 1
      }
    }
  }
}

