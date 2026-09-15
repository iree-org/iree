// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan \
// RUN:              --iree-vulkan-target=vp_android_baseline_2022 %s \
// RUN: | FileCheck %s --check-prefix=CHECK-DESCRIPTOR
//
// CHECK-DESCRIPTOR: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"
// CHECK-DESCRIPTOR-SAME: features = "spirv:v1.3,cap:Shader"
//
// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan \
// RUN:              --iree-vulkan-target=vp_android_baseline_2022 \
// RUN:              --iree-vulkan-dispatch-abi=bda %s \
// RUN: | FileCheck %s --check-prefix=CHECK-BDA
//
// CHECK-BDA: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-bda-v1"
// CHECK-BDA-SAME: cap:Int64
// CHECK-BDA-SAME: cap:PhysicalStorageBufferAddresses
// CHECK-BDA-SAME: ext:SPV_KHR_physical_storage_buffer
//
// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan \
// RUN:              --iree-vulkan-target=vp_android_baseline_2022 \
// RUN:              --iree-vulkan-dispatch-abi=all %s \
// RUN: | FileCheck %s --check-prefix=CHECK-ALL
//
// CHECK-ALL: #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-bda-v1"
// CHECK-ALL-SAME: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"
//
// Replace the default features
// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan \
// RUN:              --iree-vulkan-target=vp_android_baseline_2022 \
// RUN:              --iree-vulkan-target-features=spirv:v1.6,cap:Shader,cap:Int64,cap:PhysicalStorageBufferAddresses,ext:SPV_KHR_physical_storage_buffer %s \
// RUN: | FileCheck %s --check-prefix=CHECK-FEATURES
//
// CHECK-FEATURES: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"
// CHECK-FEATURES-SAME: features = "spirv:v1.6,cap:Shader,cap:Int64,cap:PhysicalStorageBufferAddresses,ext:SPV_KHR_physical_storage_buffer"
// CHECK-FEATURES-NOT: spirv:v1.3
//
// Explicit target combined with a features override: the target still
// selects the architecture (gfx1100), but the features string overrides the
// target's default features.
// RUN: iree-compile --compile-to=preprocessing --iree-hal-target-device=vulkan \
// RUN:              --iree-vulkan-target=rx7900xtx \
// RUN:              --iree-vulkan-target-features=spirv:v1.5,cap:Shader,cap:Int64 %s \
// RUN: | FileCheck %s --check-prefix=CHECK-TARGET-OVERRIDE
//
// CHECK-TARGET-OVERRIDE: #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"
// CHECK-TARGET-OVERRIDE-SAME: arch = "gfx1100"
// CHECK-TARGET-OVERRIDE-SAME: features = "spirv:v1.5,cap:Shader,cap:Int64"
// CHECK-TARGET-OVERRIDE-NOT: spirv:v1.6
//

module {
  util.func public @foo(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    util.return %arg0 : tensor<?xf32>
  }
}
