// RUN: iree-opt -iree-hal-transformation-pipeline -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s -check-prefix=DEFAULT
// RUN: iree-opt -iree-hal-transformation-pipeline -iree-hal-target-backends=vulkan-spirv -iree-vulkan-target-env="#vk.target_env<v1.0, r(10), [VK_KHR_storage_buffer_storage_class], {maxComputeWorkGroupInvocations = 64: i32, maxComputeWorkGroupSize = dense<[8, 8, 8]>: vector<3xi32>}>" %s | IreeFileCheck %s -check-prefix=V10

// TODO(antiagainst): Passing in lenghty strings as command-line options is not
// optimal. We should consider creating a dedicated test pass to pick up
// #vk.target_env in input assembly and convert them.

// DEFAULT: spv.target_env = #spv.target_env<V_1_3, [], [], {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[64, 4, 4]> : vector<3xi32>}>
// V10:     spv.target_env = #spv.target_env<V_1_0, [SPV_KHR_storage_buffer_storage_class], [], {max_compute_workgroup_invocations = 64 : i32, max_compute_workgroup_size = dense<8> : vector<3xi32>}>
flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = dense<[4, 1, 1]> : vector<3xi32>
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
