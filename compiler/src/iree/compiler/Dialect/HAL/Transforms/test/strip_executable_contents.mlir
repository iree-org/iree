// RUN: iree-opt --split-input-file --iree-hal-strip-executable-contents %s | FileCheck %s

// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK: hal.executable.variant public @backend
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK: hal.executable.export public @entry0
    hal.executable.export @entry0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>,
        #hal.descriptor_set.binding<1, storage_buffer>
      ]>
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
