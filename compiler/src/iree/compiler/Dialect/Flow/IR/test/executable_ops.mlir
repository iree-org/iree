// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @dispatch_ex
flow.executable @dispatch_ex {
  // CHECK: module {
  builtin.module {
    // CHECK: @dispatch0
    func.func @dispatch0() {
      return
    }
  }
  // CHECK: flow.executable.export public @dispatch0
  flow.executable.export @dispatch0
  // CHECK: flow.executable.export public @dispatch0 as("dispatch0_alias")
  flow.executable.export @dispatch0 as("dispatch0_alias")
}
