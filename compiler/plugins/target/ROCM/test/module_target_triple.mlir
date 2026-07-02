// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=gfx942 --iree-hal-dump-executable-intermediates-to=%t %s -o /dev/null
// RUN: cat %t/*.optimized.ll | FileCheck %s

// Regression test for the target triple being set on the finalized LLVM module.
//
// The ROCm target backend must set the target triple of the module, because
// the LLVM backend expects it to be set.
// See llvm-project@00a6186128d3 ("AMDGPU: Prefer getting the triple from the
// module over the TargetMachine").
// Therefore, the ROCMTargetBackend validation checks for the target triple to
// be set and emits an explicit error if that is not the case.

// CHECK: target triple = "amdgcn-amd-amdhsa"

#pipeline_layout = #hal.pipeline.layout<bindings = []>
hal.executable.source public @exe {
  hal.executable.export public @empty ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {subgroup_size = 64 : index, workgroup_size = [1 : index, 1 : index, 1 : index]}
  builtin.module {
    llvm.func @empty() attributes {rocdl.kernel} {
      llvm.return
    }
  }
}
