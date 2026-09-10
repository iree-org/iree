// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(iree-hal-serialize-target-executables{target=llvm-cpu}))' \
// RUN:   --verify-diagnostics %s -o -

// The range reaches LLVM as a `vscale_range` function attribute, so an invalid
// one has to be rejected before it can be attached.

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

// `+zvl512b` guarantees an RVV VLEN of 512 bits, i.e. a vscale of at least 8,
// which the configured maximum of 4 contradicts.
#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "generic-rv64", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl512b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 64 : index, target_triple = "riscv64-unknown-unknown-eabi-elf", vscale_range = [1, 4]}>
builtin.module {
  hal.executable public @vscale_min_exceeds_max {
    // expected-error @+2 {{invalid vscale range [8, 4]: minimum exceeds maximum; the minimum is raised to 8 by the target's RVV VLEN features}}
    // expected-error @+1 {{failed to serialize executable for target backend llvm-cpu}}
    hal.executable.variant public @embedded_elf_riscv_64 target(#executable_target_riscv64) {
      hal.executable.export public @fn ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        llvm.func @fn() {
          llvm.return
        }
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

// LLVM requires both ends of a `vscale_range` to be powers of two.
#executable_target_arm_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "generic", cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf", vscale_range = [1, 12]}>
builtin.module {
  hal.executable public @vscale_range_not_power_of_two {
    // expected-error @+2 {{'vscale_range' [1, 12] must consist of powers of two}}
    // expected-error @+1 {{failed to serialize executable for target backend llvm-cpu}}
    hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_arm_64) {
      hal.executable.export public @fn ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        llvm.func @fn() {
          llvm.return
        }
      }
    }
  }
}
