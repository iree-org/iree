// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

//      CHECK: #executable_target_vmvx_bytecode = #hal.executable.target<"vmvx-bytecode">
//      CHECK: #executable_target_vulkan_spirv0 = #hal.executable.target<"vulkan-spirv">
//      CHECK: #executable_target_vulkan_spirv1 = #hal.executable.target<"vulkan-spirv", {
// CHECK-SAME:   key = "value",
// CHECK-SAME:   property = 7 : index
// CHECK-SAME: }>

// CHECK-LABEL: executable.targets
"executable.targets"() {
  // CHECK-SAME: a = #executable_target_vmvx_bytecode,
  a = #hal.executable.target<"vmvx-bytecode">,
  // CHECK-SAME: b = #executable_target_vulkan_spirv0,
  b = #hal.executable.target<"vulkan-spirv">,
  // CHECK-SAME: c = #executable_target_vulkan_spirv1
  c = #hal.executable.target<"vulkan-spirv", {
    key = "value",
    property = 7 : index
  }>
} : () -> ()
