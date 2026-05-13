// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=gfx942 --iree-rocm-use-spirv \
// RUN:   --iree-hal-dump-executable-binaries-to=%t %s -o /dev/null
// RUN: ls %t | FileCheck %s --check-prefix=FILES --implicit-check-not=.hsaco

// Verify that ROCm serialization uses the --iree-rocm-use-spirv flag to produce
// a SPIR-V binary through the normal HAL executable source path.

// FILES: module_serialize_spirv_flag_test_rocm_spirv_fb.spv

#pipeline_layout = #hal.pipeline.layout<bindings = []>

hal.executable.source public @serialize_spirv_flag_test {
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
