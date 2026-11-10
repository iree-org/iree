// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(hal.executable(iree-hal-serialize-target-executables{target=llvm-cpu dump-intermediates-path=-}))' \
// RUN:   %s -o /dev/null | FileCheck %s

// Tests that an export without a file location gets line 0 in the source
// location table.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
builtin.module {
  hal.executable public @executable {
    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target) {
      hal.executable.export public @fn ordinal(0) layout(#pipeline_layout) loc(unknown)
      builtin.module {
        llvm.func @fn(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
          %0 = llvm.mlir.constant(0 : i32) : i32
          llvm.return %0 : i32
        }
      }
    }
  }
}

// CHECK: @iree_hal_executable_library_query_v0_source_locations = private constant [1 x %iree_hal_executable_source_location_v0_t]
// CHECK-SAME: { i32 0, i32 0, ptr
