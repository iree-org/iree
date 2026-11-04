// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(hal.executable(iree-hal-serialize-target-executables{target=llvm-cpu}))' \
// RUN:   %s | FileCheck %s

// Tests that the inline data of an executable object is written to the object
// file passed to the linker. Only the object defines @object_fn and the
// embedded linker runs with --no-undefined, so linking fails if that file is
// empty or its contents have not been flushed.

// The object data is a linker script rather than a relocatable object, so the
// test needs no binary data. It defines @object_fn as an alias of the library
// query function, which the generated object always defines.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
builtin.module {
  hal.executable public @executable {
    // CHECK: hal.executable.binary public @embedded_elf_x86_64
    // CHECK-SAME: format = "embedded-elf-x86_64"
    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target) objects([
      #hal.executable.object<{path = "object.o", data = "HIDDEN(object_fn = iree_hal_executable_library_query);"}>
    ]) {
      hal.executable.export public @fn ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        llvm.func @object_fn() attributes {sym_visibility = "private"}
        llvm.func @fn(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
          llvm.call @object_fn() : () -> ()
          %0 = llvm.mlir.constant(0 : i32) : i32
          llvm.return %0 : i32
        }
      }
    }
  }
}
