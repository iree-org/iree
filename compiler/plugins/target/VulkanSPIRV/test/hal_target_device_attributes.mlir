// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan %s \
// RUN: | FileCheck %s --check-prefix=CHECK-DESCRIPTOR
//
// CHECK-DESCRIPTOR: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"
// CHECK-DESCRIPTOR-SAME: features = "spirv:v1.3,cap:Shader"
//
// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan \
// RUN:              --iree-vulkan-dispatch-abi=bda %s \
// RUN: | FileCheck %s --check-prefix=CHECK-BDA
//
// CHECK-BDA: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-bda-v1"
// CHECK-BDA-SAME: cap:Int64
// CHECK-BDA-SAME: cap:PhysicalStorageBufferAddresses
// CHECK-BDA-SAME: ext:SPV_KHR_physical_storage_buffer
//
// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan \
// RUN:              --iree-vulkan-dispatch-abi=all %s \
// RUN: | FileCheck %s --check-prefix=CHECK-ALL
//
// CHECK-ALL: #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-bda-v1"
// CHECK-ALL-SAME: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"

module {
  util.func public @foo(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    util.return %arg0 : tensor<?xf32>
  }
}
