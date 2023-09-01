// RUN: iree-compile --split-input-file --compile-mode=vm \
// RUN: --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK: "name": "simple_module"
vm.module @simple_module {
  // CHECK: "types": [{
  // CHECK: "full_name": "f32"

  // CHECK: "exported_functions":
  // CHECK: "local_name": "func"
  vm.export @func

  // CHECK: "function_signatures":
  // CHECK-NEXT: "calling_convention": "0f_f"

  // CHECK: "function_descriptors":

  // CHECK-NEXT: {
  // CHECK-NEXT:   "bytecode_offset": 0
  // CHECK-NEXT:   "bytecode_length": 14
  // CHECK-NEXT:   "requirements": "EXT_F32"
  // CHECK-NEXT:   "reserved": 0
  // CHECK-NEXT:   "block_count": 1
  // CHECK-NEXT:   "i32_register_count": 1
  // CHECK-NEXT:   "ref_register_count": 0
  // CHECK-NEXT: }
  vm.func @func(%arg0 : f32) -> f32 {
    %0 = vm.add.f32 %arg0, %arg0 : f32
    vm.return %0 : f32
  }

  //      CHECK: "bytecode_data": [
  // CHECK-NEXT:   121,
  // CHECK-NEXT:   224,
  // CHECK-NEXT:   10,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   90,
  // CHECK-NEXT:   1,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0,
  // CHECK-NEXT:   0
  // CHECK-NEXT: ]
}
