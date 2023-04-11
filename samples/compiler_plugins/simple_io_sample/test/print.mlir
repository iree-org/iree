// RUN: iree-opt --iree-plugin=simple_io_sample --iree-print-plugin-info --pass-pipeline='builtin.module(iree-simpleio-legalize)' %s | FileCheck %s

// CHECK: func.func private @simple_io.print()
func.func @main() {
  // CHECK: call @simple_io.print
  simple_io.print
  func.return
}
