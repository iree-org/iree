  // Copyright Header: some indented comment line
  // with multiple lines preceding the RUN directives
//RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(util.func(test-pass))' \
//RUN:   %s | FileCheck %s

// CHECK-LABEL: @run_variants
util.func @run_variants() {
  // CHECK: util.return
  util.return
}
