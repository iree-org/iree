// RUN: iree-opt --split-input-file --iree-hal-resolve-export-ordinals %s | FileCheck %s

hal.executable @exe0 {
  hal.executable.variant @target target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export public @entry123 ordinal(123) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
  }
}
hal.executable @exe1 {
  hal.executable.variant @target target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export public @entry456 ordinal(456) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
  }
}

// CHECK-LABEL: @resolve
util.func public @resolve() -> (index, index) {
  // CHECK: %[[ENTRY123:.+]] = arith.constant 123
  %entry123 = hal.executable.export.ordinal target(@exe0::@target::@entry123) : index
  // CHECK: %[[ENTRY465:.+]] = arith.constant 456
  %entry456 = hal.executable.export.ordinal target(@exe1::@target::@entry456) : index
  // CHECK: util.return %[[ENTRY123]], %[[ENTRY465]]
  util.return %entry123, %entry456 : index, index
}
