// XNACK and SRAM ECC must reach LLVM as module flags and survive native
// serialization in the ELF target ID. Omitting a mode preserves Any.
// The LLVM target machine must not receive these modes as subtarget features.
// ASM: ; llc -mtriple=amdgpu9.42-amd-amdhsa -mcpu=gfx942 -mattr='-fma-mix-insts'

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=mi300x \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF-ANY
// RUN: cat %t/*.linked.ll | FileCheck %s --check-prefix=IR-ANY --implicit-check-not='!"amdgpu.xnack"' --implicit-check-not='!"amdgpu.sramecc"'
// RUN: cat %t/*.rocmasm | FileCheck %s --check-prefix=ASM
// ELF-ANY: data = dense<"0x7F454C46{{([0-9A-F]{88})}}4C050000
// IR-ANY: target triple = "amdgpu9.42-amd-amdhsa"

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=mi300x --iree-rocm-target-features=+xnack \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF-XNACKON
// RUN: cat %t/*.linked.ll | FileCheck %s --check-prefix=IR-XNACKON --implicit-check-not='!"amdgpu.sramecc"'
// RUN: cat %t/*.rocmasm | FileCheck %s --check-prefix=ASM
// ELF-XNACKON: data = dense<"0x7F454C46{{([0-9A-F]{88})}}4C070000
// IR-XNACKON: target triple = "amdgpu9.42-amd-amdhsa"
// IR-XNACKON-DAG: !{i32 1, !"amdgpu.xnack", i32 1}

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=mi300x --iree-rocm-target-features=-xnack \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF-XNACKOFF
// RUN: cat %t/*.linked.ll | FileCheck %s --check-prefix=IR-XNACKOFF --implicit-check-not='!"amdgpu.sramecc"'
// RUN: cat %t/*.rocmasm | FileCheck %s --check-prefix=ASM
// ELF-XNACKOFF: data = dense<"0x7F454C46{{([0-9A-F]{88})}}4C060000
// IR-XNACKOFF: target triple = "amdgpu9.42-amd-amdhsa"
// IR-XNACKOFF-DAG: !{i32 1, !"amdgpu.xnack", i32 0}

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=mi300x --iree-rocm-target-features=+sramecc,-xnack \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF-ECCON
// RUN: cat %t/*.linked.ll | FileCheck %s --check-prefix=IR-ECCON
// RUN: cat %t/*.rocmasm | FileCheck %s --check-prefix=ASM
// ELF-ECCON: data = dense<"0x7F454C46{{([0-9A-F]{88})}}4C0E0000
// IR-ECCON: target triple = "amdgpu9.42-amd-amdhsa"
// IR-ECCON-DAG: !{i32 1, !"amdgpu.sramecc", i32 1}
// IR-ECCON-DAG: !{i32 1, !"amdgpu.xnack", i32 0}

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=mi300x --iree-rocm-target-features=-sramecc,+xnack \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t --iree-rocm-container-type=hsaco %s | FileCheck %s --check-prefix=ELF-ECCOFF
// RUN: cat %t/*.linked.ll | FileCheck %s --check-prefix=IR-ECCOFF
// RUN: cat %t/*.rocmasm | FileCheck %s --check-prefix=ASM
// ELF-ECCOFF: data = dense<"0x7F454C46{{([0-9A-F]{88})}}4C0B0000
// IR-ECCOFF: target triple = "amdgpu9.42-amd-amdhsa"
// IR-ECCOFF-DAG: !{i32 1, !"amdgpu.sramecc", i32 0}
// IR-ECCOFF-DAG: !{i32 1, !"amdgpu.xnack", i32 1}

// RUN: not iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=mi455x --iree-rocm-target-features=-xnack %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CDNA5OFF
// CDNA5OFF: invalid ROCM target ID 'gfx1250:xnack-'

// RUN: not iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=cdna5 --iree-rocm-target-features=+xnack %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CDNA5ON
// CDNA5ON: invalid ROCM target ID 'gfx1250:xnack+'

// RUN: not iree-opt --iree-hal-transformation-pipeline --iree-hal-target-device=hip \
// RUN:   --iree-rocm-target=gfx1100 --iree-rocm-target-features=+sramecc %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALIDFEATURE
// INVALIDFEATURE: invalid ROCM target ID 'gfx1100:sramecc+'

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
