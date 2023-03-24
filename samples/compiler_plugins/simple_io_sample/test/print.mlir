// RUN: iree-opt --iree-plugin=simple_io_sample --iree-print-plugin-info %s

func.func @main() {
  // CHECK: call @simple_io.print
  simple_io.print
  func.return
}
