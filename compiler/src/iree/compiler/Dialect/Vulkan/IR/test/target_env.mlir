// Test parsing and printing Vulkan target environment attribute.

// RUN: iree-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s | FileCheck %s

"vk_configure_op"() {
  // CHECK:      #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], AMD:DiscreteGPU, {
  // CHECK-SAME:   maxComputeSharedMemorySize = 16384 : i32,
  // CHECK-SAME:   maxComputeWorkGroupInvocations = 1024 : i32,
  // CHECK-SAME:   maxComputeWorkGroupSize = dense<[128, 8, 4]> : vector<3xi32>
  // CHECK-SAME:   subgroupFeatures = 63 : i32,
  // CHECK-SAME:   subgroupSize = 4 : i32
  // CHECK-SAME: }>
  target_env = #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], AMD:DiscreteGPU, {
    maxComputeSharedMemorySize = 16384: i32,
    maxComputeWorkGroupInvocations = 1024: i32,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63: i32,
    subgroupSize = 4: i32
  }>
} : () -> ()

// -----

"vk_configure_op"() {
  //      CHECK: #vk.target_env
  // CHECK-SAME:   VK_NV_cooperative_matrix
  // CHECK-SAME:   cooperativeMatrixPropertiesNV =
  // CHECK-SAME:     aType = i8, bType = i8, cType = i32, kSize = 32 : i32,
  // CHECK-SAME:       mSize = 8 : i32, nSize = 8 : i32, resultType = i32,
  // CHECK-SAME:       scope = 3 : i32
  // CHECK-SAME:     aType = f16, bType = f16, cType = f16, kSize = 16 : i32,
  // CHECK-SAME:       mSize = 8 : i32, nSize = 8 : i32, resultType = f16,
  // CHECK-SAME:       scope = 3 : i32
  target_env =
    #vk.target_env<v1.2, r(133),
      [VK_KHR_storage_buffer_storage_class, VK_NV_cooperative_matrix],
      NVIDIA:DiscreteGPU,
      {maxComputeSharedMemorySize = 49152: i32,
       maxComputeWorkGroupInvocations = 1024: i32,
       maxComputeWorkGroupSize = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
       subgroupFeatures = 63: i32, subgroupSize = 32: i32,
       cooperativeMatrixPropertiesNV = [
         {mSize = 8: i32, nSize = 8: i32, kSize = 32: i32, aType = i8,
          bType = i8, cType = i32, resultType = i32, scope = 3: i32},
         {mSize = 8: i32, nSize = 8: i32, kSize = 16: i32, aType = f16,
          bType = f16, cType = f16, resultType = f16, scope = 3: i32}]
      }>
} : () -> ()

// -----

"vk_configure_op"() {
  // CHECK:      Qualcomm:IntegratedGPU:100925441
  // CHECK-SAME: shaderFloat64
  // CHECK-SAME: shaderInt16
  target_env = #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], Qualcomm:IntegratedGPU:0x6040001, {
    maxComputeSharedMemorySize = 16384: i32,
    maxComputeWorkGroupInvocations = 1024: i32,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63: i32,
    subgroupSize = 4: i32,
    shaderFloat64, shaderInt16
  }>
} : () -> ()

// -----

"unknown_vulkan_version"() {
  // expected-error @+1 {{unknown Vulkan version: v10.8}}
  target_env = #vk.target_env<v10.8, r(0), [], {
    maxComputeWorkGroupInvocations = 128: i32,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>
  }>
} : () -> ()

// -----

"unknown_vulkan_extension"() {
  // expected-error @+1 {{unknown Vulkan extension: VK_KHR_something}}
  target_env = #vk.target_env<v1.0, r(10), [VK_KHR_something], {
    maxComputeWorkGroupInvocations = 128: i32,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>
  }>
} : () -> ()

// -----

"wrong_vendor_id"() {
  // expected-error @+1 {{unknown vendor: AVendor}}
  target_env = #vk.target_env<v1.0, r(10), [], AVendor:Unknown, {
    maxComputeSharedMemorySize = 16384: i32,
    maxComputeWorkGroupInvocations = 1024: i32,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63: i32,
    subgroupSize = 4: i32
  }>
} : () -> ()

// -----

"wrong_device_type"() {
  // expected-error @+1 {{unknown device type: ADeviceType}}
  target_env = #vk.target_env<v1.0, r(10), [], NVIDIA:ADeviceType, {
    maxComputeSharedMemorySize = 16384: i32,
    maxComputeWorkGroupInvocations = 1024: i32,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 63: i32,
    subgroupSize = 4: i32
  }>
} : () -> ()

// -----

"missing_core_1_1_properties_field"() {
  // expected-error @+1 {{capabilities must be a vulkan::CapabilitiesAttr dictionary attribute}}
  target_env = #vk.target_env<v1.0, r(10), [], Unknown:Unknown, {
    maxComputeWorkGroupInvocations = 128: i32
  }>
} : () -> ()

// -----

"unknown_core_1_1_properties_field"() {
  // expected-error @+1 {{capabilities must be a vulkan::CapabilitiesAttr dictionary attribute}}
  target_env = #vk.target_env<v1.0, r(10), [], Unknown:Unknown, {
    maxComputeWorkGroupInvocations = 128: i32,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>,
    moreStuff = 8: i32
  }>
} : () -> ()

// -----

"wrong_subgroup_bit"() {
  // expected-error @+1 {{capabilities must be a vulkan::CapabilitiesAttr dictionary attribute}}
  target_env = #vk.target_env<v1.0, r(10), [], Unknown:Unknown, {
    maxComputeSharedMemorySize = 16384: i32,
    maxComputeWorkGroupInvocations = 1024: i32,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>,
    subgroupFeatures = 0xffffffff: i32,
    subgroupSize = 4: i32
  }>
} : () -> ()
