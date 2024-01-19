// RUN: iree-compile --split-input-file --compile-mode=vm \
// RUN: --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK-LABEL: simple_module
// CHECK: "attrs":
// CHECK:   "key": "module_attr_0"
// CHECK:   "value": "MODULE_ATTR_0"
// CHECK:   "key": "module_attr_1"
// CHECK:   "value": "MODULE_ATTR_1"
vm.module @simple_module attributes {
  iree.reflection = {
    module_attr_0 = "MODULE_ATTR_0",
    module_attr_1 = "MODULE_ATTR_1"
  }
} {
  vm.export @func
  // CHECK: "exported_functions":
  // CHECK: "attrs":
  // CHECK:   "key": "func_attr_0"
  // CHECK:   "value": "FUNC_ATTR_0"
  // CHECK:   "key": "func_attr_1"
  // CHECK:   "value": "FUNC_ATTR_1"
  vm.func @func(%arg0 : i32) -> i32 attributes {
    iree.reflection = {
      func_attr_0 = "FUNC_ATTR_0",
      func_attr_1 = "FUNC_ATTR_1"
    }
  } {
    vm.return %arg0 : i32
  }
}
