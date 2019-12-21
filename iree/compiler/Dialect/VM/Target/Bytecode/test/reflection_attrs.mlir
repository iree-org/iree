// RUN: iree-translate -split-input-file -iree-vm-ir-to-bytecode-module -iree-vm-bytecode-module-output-format=flatbuffer-text %s | IreeFileCheck %s

// CHECK-LABEL: simple_module
vm.module @simple_module {
  vm.export @func
  // CHECK: reflection_attrs:
  // CHECK:   key: "f"
  // CHECK:   value: "FOOBAR"
  // CHECK:   key: "fv"
  // CHECK:   value: "INFINITY"
  vm.func @func(%arg0 : i32) -> i32
    attributes { iree.reflection = { f = "FOOBAR", fv = "INFINITY" } }
  {
    vm.return %arg0 : i32
  }
}
