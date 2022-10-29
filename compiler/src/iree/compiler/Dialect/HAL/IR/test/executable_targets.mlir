// RUN: iree-opt --allow-unregistered-dialect --split-input-file %s | iree-opt --allow-unregistered-dialect --split-input-file | FileCheck %s

//      CHECK: #executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
//      CHECK: #executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">
//      CHECK: #executable_target_vulkan_spirv_fb1 = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
// CHECK-SAME:   key = "value",
// CHECK-SAME:   property = 7 : index
// CHECK-SAME: }>

// CHECK-LABEL: executable.targets
"executable.targets"() {
  // CHECK-SAME: a = #executable_target_vmvx_bytecode_fb,
  a = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">,
  // CHECK-SAME: b = #executable_target_vulkan_spirv_fb,
  b = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">,
  // CHECK-SAME: c = #executable_target_vulkan_spirv_fb1
  c = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
    key = "value",
    property = 7 : index
  }>
} : () -> ()
