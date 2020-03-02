// Test parsing and printing Vulkan target environment attribute.

// RUN: iree-opt -split-input-file -verify-diagnostics %s | IreeFileCheck %s

"vk_configure_op"() {
  // CHECK:      #vk.target_env<v1.0, r(0), [], {
  // CHECK-SAME:   maxComputeWorkGroupInvocations = 128 : i32,
  // CHECK-SAME:   maxComputeWorkGroupSize = dense<[64, 4, 4]> : vector<3xi32>
  // CHECK-SAME: }>
  target_env = #vk.target_env<v1.0, r(0), [], {
    maxComputeWorkGroupInvocations = 128: i32,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>
  }>
} : () -> ()


"vk_configure_op"() {
  // CHECK:      #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], {
  // CHECK-SAME:   maxComputeWorkGroupInvocations = 1024 : i32,
  // CHECK-SAME:   maxComputeWorkGroupSize = dense<[128, 8, 4]> : vector<3xi32>
  // CHECK-SAME: }>
  target_env = #vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], {
    maxComputeWorkGroupInvocations = 1024: i32,
    maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>
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

"missing_core_1_1_properties_field"() {
  // expected-error @+1 {{core10Properties must be a vulkan::Core10PropertiesAttr dictionary attribute}}
  target_env = #vk.target_env<v1.0, r(10), [], {
    maxComputeWorkGroupInvocations = 128: i32
  }>
} : () -> ()

// -----

"unknown_core_1_1_properties_field"() {
  // expected-error @+1 {{core10Properties must be a vulkan::Core10PropertiesAttr dictionary attribute}}
  target_env = #vk.target_env<v1.0, r(10), [], {
    maxComputeWorkGroupInvocations = 128: i32,
    maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>,
    moreStuff = 8: i32
  }>
} : () -> ()
