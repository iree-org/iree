// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=valhall %s | FileCheck %s --check-prefixes=VALHALL,VALHALL1
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=valhall1 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL1
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g77 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL1
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g57 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL1
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=valhall2 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL2
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g78 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL2
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=valhall3 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL3
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g710 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL3
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g510 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL3
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g310 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL3
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=valhall4 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL4
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g715 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL4
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g615 %s | FileCheck %s --check-prefixes=VALHALL,VALHALL4

// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=arm5gen %s | FileCheck %s --check-prefixes=ARM5GEN
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g720 %s | FileCheck %s --check-prefixes=ARM5GEN
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g725 %s | FileCheck %s --check-prefixes=ARM5GEN
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-vulkan-target=mali-g1 %s | FileCheck %s --check-prefixes=ARM5GEN

// VALHALL1: target_info = #iree_gpu.target<arch = "valhall1",
// VALHALL2: target_info = #iree_gpu.target<arch = "valhall2",
// VALHALL3: target_info = #iree_gpu.target<arch = "valhall3",
// VALHALL4: target_info = #iree_gpu.target<arch = "valhall4",
// VALHALL-SAME: features = "spirv:v1.6,cap:Shader"
// VALHALL-SAME: wgp = <compute =
// VALHALL-SAME: fp32|fp16|int32|int16|int8
// VALHALL-SAME: storage =
// VALHALL-SAME: b64|b32|b16|b8
// VALHALL-SAME: subgroup =
// VALHALL-SAME: shuffle|arithmetic
// VALHALL-SAME: dot =
// VALHALL-SAME: dp4xi8toi32
// VALHALL-SAME: subgroup_size_choices = [16]
// VALHALL-SAME: max_workgroup_sizes = [512, 512, 512]
// VALHALL-SAME: max_thread_count_per_workgroup = 512
// VALHALL-SAME: max_workgroup_memory_bytes = 32768
// VALHALL-SAME: max_workgroup_counts = [65535, 65535, 65535]

// ARM5GEN: target_info = #iree_gpu.target<arch = "arm5gen",
// ARM5GEN-SAME: features = "spirv:v1.6,cap:Shader"
// ARM5GEN-SAME: wgp = <compute =
// ARM5GEN-SAME: fp64|fp32|fp16|int64|int32|int16|int8
// ARM5GEN-SAME: storage =
// ARM5GEN-SAME: b64|b32|b16|b8
// ARM5GEN-SAME: subgroup =
// ARM5GEN-SAME: shuffle|arithmetic
// ARM5GEN-SAME: dot =
// ARM5GEN-SAME: dp4xi8toi32
// ARM5GEN-SAME: subgroup_size_choices = [16]
// ARM5GEN-SAME: max_workgroup_sizes = [1024, 1024, 1024]
// ARM5GEN-SAME: max_thread_count_per_workgroup = 1024
// ARM5GEN-SAME: max_workgroup_memory_bytes = 32768
// ARM5GEN-SAME: max_workgroup_counts = [65535, 65535, 65535]

stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch() {
      return
    }
  }
}
