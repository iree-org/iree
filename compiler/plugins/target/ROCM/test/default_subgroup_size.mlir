// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=gfx1250 --iree-hal-dump-executable-intermediates-to=%t %s -o /dev/null
// RUN: cat %t/*.rocmasm | FileCheck %s

// Verify that serialization uses the target's preferred subgroup size when an
// export does not specify one. gfx1250 only supports wave32; asking LLVM for
// wave64 silently produces an executable with no kernels.

// CHECK: .globl empty
// CHECK: amdhsa.kernels:
// CHECK: .name: empty

#pipeline_layout = #hal.pipeline.layout<bindings = []>
hal.executable.source public @exe {
  hal.executable.export public @empty ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
  builtin.module {
    llvm.func @empty() attributes {rocdl.kernel} {
      llvm.return
    }
  }
}
