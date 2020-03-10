// Tests printing and parsing of executable/structural ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @dispatch_ex
flow.executable @dispatch_ex {
  // CHECK: module {
  module {
    // CHECK: @dispatch0
    func @dispatch0() {
      return
    }
  }
  // CHECK: flow.dispatch.entry @dispatch0
  flow.dispatch.entry @dispatch0
  // CHECK: flow.dispatch.entry @dispatch0 as("dispatch0_alias")
  flow.dispatch.entry @dispatch0 as("dispatch0_alias")
}
