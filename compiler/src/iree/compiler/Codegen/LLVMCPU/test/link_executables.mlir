// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-llvmcpu-link-executables{target=llvm-cpu})" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-llvmcpu-link-executables{target=llvm-cpu},iree-vmvx-link-executables)" %s | FileCheck %s --check-prefix=BOTH

// An executable whose variants all target llvm-cpu is linked and every
// reference to it (root, variant and export) is redirected to the linked
// executable.

#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @embedded_elf_x86_64 target(#cpu_target) {
    hal.executable.export public @dispatch_0 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_0() {
        llvm.return
      }
    }
  }
}
hal.executable private @dispatch_1 {
  hal.executable.variant @embedded_elf_x86_64 target(#cpu_target) {
    hal.executable.export public @dispatch_1 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_1() {
        llvm.return
      }
    }
  }
}
func.func @cpu_only() {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_0) : !hal.executable
  %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_0::@embedded_elf_x86_64::@dispatch_0) : i64
  return
}

// The linked executable is inserted first; no source executable may follow it.
// CHECK:       hal.executable private @link_executables_linked {
// CHECK-NEXT:    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64
// CHECK:           hal.executable.export public @dispatch_0 ordinal(0)
// CHECK:           hal.executable.export public @dispatch_1 ordinal(1)
// CHECK-NOT:   hal.executable private
// CHECK-LABEL: func.func @cpu_only
// CHECK:         hal.executable.lookup device(%{{.+}} : !hal.device) executable(@link_executables_linked)
// CHECK:         function(@link_executables_linked::@embedded_elf_x86_64::@dispatch_0)

// -----

// A local device with both an llvm-cpu and a vmvx backend: every executable
// gets a variant per backend and dispatches select one at runtime, with one
// lookup per variant. Each backend links its own variants in turn, so after
// the llvm-cpu linker @dispatch_shared survives with its vmvx variant and only
// the llvm-cpu lookup may move to the linked executable. Once the vmvx linker
// has moved the remaining variant the vmvx lookup follows it.

#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @dispatch_shared {
  hal.executable.variant @embedded_elf_x86_64 target(#cpu_target) {
    hal.executable.export public @dispatch_shared ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_shared() {
        llvm.return
      }
    }
  }
  hal.executable.variant @vmvx_bytecode_fb target(#vmvx_target) {
    hal.executable.export public @dispatch_shared ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      vm.module @module {
        vm.func @dispatch_shared() {
          vm.return
        }
        vm.export @dispatch_shared
      }
    }
  }
}
hal.executable private @dispatch_cpu {
  hal.executable.variant @embedded_elf_x86_64 target(#cpu_target) {
    hal.executable.export public @dispatch_cpu ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_cpu() {
        llvm.return
      }
    }
  }
}
hal.executable private @dispatch_vmvx {
  hal.executable.variant @vmvx_bytecode_fb target(#vmvx_target) {
    hal.executable.export public @dispatch_vmvx ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      vm.module @module {
        vm.func @dispatch_vmvx() {
          vm.return
        }
        vm.export @dispatch_vmvx
      }
    }
  }
}
func.func @multi_backend(%variant: index) {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  scf.index_switch %variant
  case 0 {
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_shared) : !hal.executable
    %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_shared::@embedded_elf_x86_64::@dispatch_shared) : i64
    scf.yield
  }
  case 1 {
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_shared) : !hal.executable
    %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_shared::@vmvx_bytecode_fb::@dispatch_shared) : i64
    scf.yield
  }
  default {
    scf.yield
  }
  %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_vmvx) : !hal.executable
  %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_vmvx::@vmvx_bytecode_fb::@dispatch_vmvx) : i64
  return
}

// CHECK-NOT:   hal.executable private @dispatch_cpu
// CHECK:       hal.executable private @link_executables_linked {
// CHECK-NEXT:    hal.executable.variant public @embedded_elf_x86_64_0 target(#executable_target_embedded_elf_x86_64
// CHECK:           hal.executable.export public @dispatch_shared ordinal(0)
// CHECK:           hal.executable.export public @dispatch_cpu ordinal(1)
// CHECK:       hal.executable private @dispatch_shared {
// CHECK-NOT:     hal.executable.variant public @embedded_elf_x86_64
// CHECK:         hal.executable.variant public @vmvx_bytecode_fb
// CHECK-LABEL: func.func @multi_backend
// CHECK:         case 0 {
// CHECK:           hal.executable.lookup device(%{{.+}} : !hal.device) executable(@link_executables_linked)
// CHECK:           function(@link_executables_linked::@embedded_elf_x86_64_0::@dispatch_shared)
// CHECK:         case 1 {
// CHECK:           hal.executable.lookup device(%{{.+}} : !hal.device) executable(@dispatch_shared)
// CHECK:           function(@dispatch_shared::@vmvx_bytecode_fb::@dispatch_shared)
// CHECK:         hal.executable.lookup device(%{{.+}} : !hal.device) executable(@dispatch_vmvx)

// BOTH-NOT:    hal.executable private @dispatch_shared
// BOTH-NOT:    hal.executable private @dispatch_vmvx
// BOTH-LABEL:  func.func @multi_backend
// BOTH:          case 0 {
// BOTH:            hal.executable.lookup device(%{{.+}} : !hal.device) executable(@link_executables_linked)
// BOTH:            function(@link_executables_linked::@embedded_elf_x86_64_0::@dispatch_shared)
// BOTH:          case 1 {
// BOTH:            hal.executable.lookup device(%{{.+}} : !hal.device) executable(@link_executables_linked_vmvx)
// BOTH:            function(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_shared)
// BOTH:          hal.executable.lookup device(%{{.+}} : !hal.device) executable(@link_executables_linked_vmvx)
// BOTH:          function(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_vmvx)

// -----

// Multi-versioned CPU executables on one device: each llvm-cpu target is
// linked by its own call. The first call only links half of the variants and
// retargets the lookup of its switch case; the second call links the rest.
// Both lookups end up on the linked executable holding both variants. As in
// conversion output each switch case has its own lookup; case 1 dispatches
// indirectly to cover both dispatch forms consuming a lookup.

#x86_avx2 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2"}>
#x86_avx512 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f"}>
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @dispatch_mv_0 {
  hal.executable.variant @avx2 target(#x86_avx2) {
    hal.executable.export public @dispatch_mv_0 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_mv_0() {
        llvm.return
      }
    }
  }
  hal.executable.variant @avx512 target(#x86_avx512) {
    hal.executable.export public @dispatch_mv_0 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_mv_0() {
        llvm.return
      }
    }
  }
}
hal.executable private @dispatch_mv_1 {
  hal.executable.variant @avx2 target(#x86_avx2) {
    hal.executable.export public @dispatch_mv_1 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_mv_1() {
        llvm.return
      }
    }
  }
  hal.executable.variant @avx512 target(#x86_avx512) {
    hal.executable.export public @dispatch_mv_1 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @dispatch_mv_1() {
        llvm.return
      }
    }
  }
}
func.func @multi_versioned(%variant: index, %cmd: !hal.command_buffer, %buffer: !hal.buffer) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0_i32 = arith.constant 0 : i32
  %device = hal.command_buffer.device<%cmd : !hal.command_buffer> : !hal.device
  scf.index_switch %variant
  case 0 {
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_mv_0) : !hal.executable
    %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_mv_0::@avx2::@dispatch_mv_0) : i64
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%fn] workgroups([%c1, %c1, %c1]) constants([%c0_i32]) bindings([
      (%buffer : !hal.buffer)[%c0, %c128]
    ]) flags("None")
    scf.yield
  }
  case 1 {
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_mv_0) : !hal.executable
    %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_mv_0::@avx512::@dispatch_mv_0) : i64
    hal.command_buffer.dispatch.indirect<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%fn] workgroups(%buffer : !hal.buffer)[%c0] constants([%c0_i32]) bindings([
      (%buffer : !hal.buffer)[%c0, %c128]
    ]) flags("None")
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// CHECK:       hal.executable private @link_executables_linked {
// CHECK:         hal.executable.variant public @embedded_elf_x86_64_0 target(
// CHECK:           hal.executable.export public @dispatch_mv_0 ordinal(0)
// CHECK:           hal.executable.export public @dispatch_mv_1 ordinal(1)
// CHECK:         hal.executable.variant public @embedded_elf_x86_64_1 target(
// CHECK:           hal.executable.export public @dispatch_mv_0 ordinal(0)
// CHECK:           hal.executable.export public @dispatch_mv_1 ordinal(1)
// CHECK-NOT:   hal.executable private
// CHECK-LABEL: func.func @multi_versioned
// CHECK:         case 0 {
// CHECK:           %[[EXE_0:.+]] = hal.executable.lookup device(%{{.+}} : !hal.device) executable(@link_executables_linked)
// CHECK:           %[[FN_0:.+]] = hal.executable.lookup.function target(%[[EXE_0]] : !hal.executable) function(@link_executables_linked::@embedded_elf_x86_64_0::@dispatch_mv_0)
// CHECK:           hal.command_buffer.dispatch<%{{.+}} : !hal.command_buffer> target(%[[EXE_0]] : !hal.executable)[%[[FN_0]]]
// CHECK:         case 1 {
// CHECK:           %[[EXE_1:.+]] = hal.executable.lookup device(%{{.+}} : !hal.device) executable(@link_executables_linked)
// CHECK:           %[[FN_1:.+]] = hal.executable.lookup.function target(%[[EXE_1]] : !hal.executable) function(@link_executables_linked::@embedded_elf_x86_64_1::@dispatch_mv_0)
// CHECK:           hal.command_buffer.dispatch.indirect<%{{.+}} : !hal.command_buffer> target(%[[EXE_1]] : !hal.executable)[%[[FN_1]]]
