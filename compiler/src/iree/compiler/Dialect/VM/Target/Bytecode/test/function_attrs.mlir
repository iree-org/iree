// RUN: iree-compile --split-input-file --compile-mode=vm \
// RUN: --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK-LABEL: simple_module
vm.module @simple_module {
  vm.export @func
  // CHECK: "exported_functions":
  // CHECK: "attrs":
  // CHECK:   "key": "f"
  // CHECK:   "value": "FOOBAR"
  // CHECK:   "key": "fv"
  // CHECK:   "value": "INFINITY"
  vm.func @func(%arg0 : i32) -> i32
    attributes { iree.reflection = { f = "FOOBAR", fv = "INFINITY" } }
  {
    vm.return %arg0 : i32
  }
}
