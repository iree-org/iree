// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-compile --compile-mode=hal-executable --iree-rocm-target=gfx942 \
// RUN:   --iree-hal-dump-executable-binaries-to=%t %s -o /dev/null
// RUN: ls %t | FileCheck %s --check-prefix=FILES --implicit-check-not=.hsaco

// Verify that ROCm serialization derives SPIR-V mode from the target attr
// format. This intentionally does not pass --iree-rocm-use-spirv.
// This should not be possible to encounter from normal iree CLI entry points,
// but preconfigured or programmatically created HAL executable variants should
// still be serialized according to their target attr.

// FILES: module_serialize_spirv_attr_test_rocm_spirv_fb.spv

#pipeline_layout = #hal.pipeline.layout<bindings = []>
#target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = none,
  subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>

hal.executable public @serialize_spirv_attr_test {
  hal.executable.variant public @rocm_spirv_fb target(<"rocm", "rocm-spirv-fb", {
    iree_codegen.target_info = #target,
    ukernels = "none"
  }>) {
    hal.executable.export public @empty ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    } attributes {subgroup_size = 64 : index, workgroup_size = [1 : index, 1 : index, 1 : index]}
    builtin.module attributes {
      llvm.data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n32:64-S32-G1-P4-A0",
      llvm.target_triple = "spirv64-amd-amdhsa"
    } {
      llvm.func spir_kernelcc @empty() attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
        llvm.return
      }
    }
  }
}
