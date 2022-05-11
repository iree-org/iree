// RUN: iree-translate --split-input-file --iree-vm-ir-to-bytecode-module --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK: "name": "simple_module"
vm.module @simple_module {
  // CHECK: "types": [{
  // CHECK: "full_name": "i32"

  // CHECK: "exported_functions":
  // CHECK: "local_name": "func"
  vm.export @func

  // CHECK: "function_descriptors":

  // CHECK-NEXT: {
  // CHECK-NEXT:   "bytecode_offset": 0
  // CHECK-NEXT:   "bytecode_length": 8
  // CHECK-NEXT:   "i32_register_count": 1
  // CHECK-NEXT:   "ref_register_count": 0
  // CHECK-NEXT: }
  vm.func @func(%arg0 : i32) -> i32 {
    vm.return %arg0 : i32
  }

  //      CHECK: "bytecode_data": [
  // CHECK-NEXT:   84,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   1,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0
  // CHECK-NEXT: ]
}
