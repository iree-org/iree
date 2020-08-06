// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s -check-prefix=DEFAULT
// RUN: iree-opt -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false}' -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-env='#vk.target_env<v1.0, r(10), [VK_KHR_shader_float16_int8, VK_KHR_storage_buffer_storage_class], {maxComputeWorkGroupInvocations = 64: i32, maxComputeWorkGroupSize = dense<[8, 8, 8]>: vector<3xi32>, shaderFloat64, shaderInt8}>' %s | IreeFileCheck %s -check-prefix=V10

// TODO(antiagainst): Passing in lenghty strings as command-line options is not
// optimal. We should consider creating a dedicated test pass to pick up
// #vk.target_env in input assembly and convert them.

// DEFAULT: spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>}
// V10:     spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, Float64, Int8], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 64 : i32, max_compute_workgroup_size = dense<8> : vector<3xi32>}>
flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = 4 : index
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
