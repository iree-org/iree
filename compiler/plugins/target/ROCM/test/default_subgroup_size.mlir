// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=gfx1250 --iree-hal-dump-executable-intermediates-to=%t \
// RUN:   --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF
// RUN: cat %t/*.rocmasm | FileCheck %s
// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=cdna5 --iree-hal-dump-executable-intermediates-to=%t \
// RUN:   --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF
// RUN: cat %t/*.rocmasm | FileCheck %s
// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=mi455x --iree-hal-dump-executable-intermediates-to=%t \
// RUN:   --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF
// RUN: cat %t/*.rocmasm | FileCheck %s

// Verify that serialization uses the target's preferred subgroup size when an
// export does not specify one. CDNA5 only supports wave32; asking LLVM for
// wave64 silently produces an executable with no kernels.

// The ELF target ID must not require selectable XNACK on CDNA5. Check the
// little-endian e_flags at byte offset 48: gfx1250 (0x49), SRAMECC any
// (0x400), and XNACK unsupported (0x0).
// ELF: data = dense<"0x7F454C46{{([0-9A-F]{88})}}49040000

// Targets with selectable XNACK must retain their feature bits: gfx942
// (0x4c), SRAMECC any (0x400), and XNACK any (0x100).
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=gfx942 --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF-CDNA3
// ELF-CDNA3: data = dense<"0x7F454C46{{([0-9A-F]{88})}}4C050000

// CHECK: .amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
// CHECK: .globl empty
// CHECK: amdhsa.kernels:
// CHECK: .name: empty
// CHECK: .wavefront_size: 32

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
