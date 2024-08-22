// RUN: iree-opt --split-input-file --iree-hal-strip-executable-contents %s | FileCheck %s

// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK: hal.executable.variant public @backend
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK: hal.executable.export public @entry0
    hal.executable.export @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>)
    // CHECK-NOT: builtin.module
    builtin.module {
      // CHECK-NOT: func.func @entry0
      func.func @entry0() {
        return
      }
    }
  }
}
