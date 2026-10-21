// RUN: iree-opt --split-input-file --verify-diagnostics --allow-unregistered-dialect --pass-pipeline="builtin.module(iree-llvmcpu-link-executables{target=llvm-cpu})" %s

// A lookup of a multi-target executable resolving functions of both its
// llvm-cpu and vmvx variants cannot load a single executable after only the
// llvm-cpu variant is linked.

#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
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
func.func @mixed_lookup() {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  // expected-error @+1 {{resolves functions from multiple executables after linking}}
  %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_shared) : !hal.executable
  %fn_cpu = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_shared::@embedded_elf_x86_64::@dispatch_shared) : i64
  %fn_vmvx = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_shared::@vmvx_bytecode_fb::@dispatch_shared) : i64
  return
}

// -----

// A looked-up executable that escapes the lookup site may have functions
// resolved elsewhere, so the variant it needs cannot be determined.

#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
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
func.func @forwarded_lookup() -> !hal.executable {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  // expected-error @+1 {{is used by 'func.return'; cannot determine which variant it resolves to}}
  %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_shared) : !hal.executable
  %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_shared::@embedded_elf_x86_64::@dispatch_shared) : i64
  return %exe : !hal.executable
}

// -----

// Root references other than lookups cannot be retargeted per site.

#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
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
// expected-error @+1 {{references partially linked executable @dispatch_shared; only hal.executable.lookup root references can be retargeted}}
func.func @root_reference() attributes {executable_ref = @dispatch_shared} {
  return
}

// -----

// Symbol uses below an unregistered op with a region cannot be enumerated, so
// a lookup inside it could not be retargeted.

#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// expected-error @+1 {{has uses that cannot be enumerated}}
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
func.func @unknown_region() {
  "unknown.region"() ({
    %c0 = arith.constant 0 : index
    %device = hal.devices.get %c0 : !hal.device
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@dispatch_shared) : !hal.executable
    %fn = hal.executable.lookup.function target(%exe : !hal.executable) function(@dispatch_shared::@embedded_elf_x86_64::@dispatch_shared) : i64
    "unknown.yield"() : () -> ()
  }) : () -> ()
  return
}
