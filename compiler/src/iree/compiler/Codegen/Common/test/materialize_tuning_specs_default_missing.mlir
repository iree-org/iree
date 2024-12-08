// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-enable-default-tuning-specs --no-implicit-module %s \
// RUN:   | FileCheck %s

// Check that we do not materialize any tuning spec when there's no default spec for the given
// target (since we do not set any target)
// CHECK:        module {
// CHECK-LABEL:    func.func @main_0

module {
  func.func @main_0() {
    return
  }
}
