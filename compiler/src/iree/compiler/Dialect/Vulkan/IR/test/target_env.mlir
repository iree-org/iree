// Test parsing and printing Vulkan target environment attribute.

// RUN: iree-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s | FileCheck %s

"vk_configure_op"() {
  // CHECK:      #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], AMD:DiscreteGPU, #vk.caps<
  // CHECK-SAME:   maxComputeSharedMemorySize = 16384,
  // CHECK-SAME:   maxComputeWorkGroupInvocations = 1024,
  // CHECK-SAME:   maxComputeWorkGroupSize = dense<[128, 8, 4]> : vector<3xi32>
  // CHECK-SAME:   subgroupFeatures = 63 : i32,
  // CHECK-SAME:   subgroupSize = 4
  // CHECK-SAME: >>
  target_env = #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], AMD:DiscreteGPU, #vk.caps<
    maxComputeSharedMemorySize = 16384,
    maxComputeWorkGroupInvocations = 1024,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63 : i32,
    subgroupSize = 4
  >>
} : () -> ()

// -----

"vk_configure_op"() {
  //      CHECK: #vk.target_env
  // CHECK-SAME:   VK_KHR_cooperative_matrix
  // CHECK-SAME:   cooperativeMatrixPropertiesKHR =
  // CHECK-SAME:     #vk.coop_matrix_props<mSize = 8, nSize = 8, kSize = 32,
  // CHECK-SAME:       aType = i8, bType = i8, cType = i32, resultType = i32,
  // CHECK-SAME:       accSat = false, scope = <Subgroup>>
  // CHECK-SAME:     #vk.coop_matrix_props<mSize = 8, nSize = 8, kSize = 16,
  // CHECK-SAME:       aType = f16, bType = f16, cType = f16, resultType = f16,
  // CHECK-SAME:       accSat = false, scope = <Subgroup>>
  target_env =
    #vk.target_env<v1.2, r(133),
      [VK_KHR_storage_buffer_storage_class, VK_KHR_cooperative_matrix],
      NVIDIA:DiscreteGPU,
      #vk.caps<maxComputeSharedMemorySize = 49152,
       maxComputeWorkGroupInvocations = 1024,
       maxComputeWorkGroupSize = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
       subgroupFeatures = 63: i32, subgroupSize = 32,
       cooperativeMatrixPropertiesKHR = [
         #vk.coop_matrix_props<
           mSize = 8, nSize = 8, kSize = 32,
           aType = i8, bType = i8, cType = i32, resultType = i32,
           accSat = false, scope = #vk.scope<Subgroup>>,
         #vk.coop_matrix_props<
           mSize = 8, nSize = 8, kSize = 16,
           aType = f16, bType = f16, cType = f16, resultType = f16,
           accSat = false, scope = #vk.scope<Subgroup>>
       ]
      >>
} : () -> ()

// -----

"vk_configure_op"() {
  // CHECK:      Qualcomm:IntegratedGPU:100925441
  // CHECK-SAME: shaderFloat64
  // CHECK-SAME: shaderInt16
  target_env = #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], Qualcomm:IntegratedGPU:0x6040001, #vk.caps<
    maxComputeSharedMemorySize = 16384,
    maxComputeWorkGroupInvocations = 1024,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63: i32,
    subgroupSize = 4,
    shaderFloat64 = unit, shaderInt16 = unit
  >>
} : () -> ()

// -----

"unknown_vulkan_version"() {
  // expected-error @+1 {{unknown Vulkan version: v10.8}}
  target_env = #vk.target_env<v10.8, r(0), [], #vk.caps<
    maxComputeWorkGroupInvocations = 128,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>
  >>
} : () -> ()

// -----

"unknown_vulkan_extension"() {
  // expected-error @+1 {{unknown Vulkan extension: VK_KHR_something}}
  target_env = #vk.target_env<v1.0, r(10), [VK_KHR_something], #vk.caps<
    maxComputeWorkGroupInvocations = 128,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>
  >>
} : () -> ()

// -----

"wrong_vendor_id"() {
  // expected-error @+1 {{unknown vendor: AVendor}}
  target_env = #vk.target_env<v1.0, r(10), [], AVendor:Unknown, #vk.caps<
    maxComputeSharedMemorySize = 16384,
    maxComputeWorkGroupInvocations = 1024,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63: i32,
    subgroupSize = 4
  >>
} : () -> ()

// -----

"wrong_device_type"() {
  // expected-error @+1 {{unknown device type: ADeviceType}}
  target_env = #vk.target_env<v1.0, r(10), [], NVIDIA:ADeviceType, #vk.caps<
    maxComputeSharedMemorySize = 16384,
    maxComputeWorkGroupInvocations = 1024,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63: i32,
    subgroupSize = 4
  >>
} : () -> ()

// -----

"missing_core_1_1_properties_field"() {
  target_env = #vk.target_env<v1.0, r(10), [], Unknown:Unknown, #vk.caps<
    maxComputeWorkGroupInvocations = 128
  // expected-error @+1 {{struct is missing required parameter: maxComputeSharedMemorySize}}
  >>
} : () -> ()

// -----

"unknown_core_1_1_properties_field"() {
  target_env = #vk.target_env<v1.0, r(10), [], Unknown:Unknown, #vk.caps<
    maxComputeSharedMemorySize = 16384,
    maxComputeWorkGroupInvocations = 128,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>,
    // expected-error @+1 {{duplicate or unknown struct parameter name: moreStuff}}
    moreStuff = 8: i32
  >>
} : () -> ()

// -----

"wrong_subgroup_bit"() {
  target_env = #vk.target_env<v1.0, r(10), [], Unknown:Unknown, #vk.caps<
    maxComputeSharedMemorySize = 16384,
    maxComputeWorkGroupInvocations = 1024,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    // expected-error @+2 {{invalid kind of attribute specified}}
    // expected-error @+1 {{failed to parse VK_CapabilitiesAttr parameter 'subgroupFeatures'}}
    subgroupFeatures = 0xffffffff: i32,
    subgroupSize = 4
  >>
} : () -> ()
