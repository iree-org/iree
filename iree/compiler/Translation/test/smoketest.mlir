// RUN: iree-translate -split-input-file -iree-mlir-to-vm-bytecode-module -iree-vm-bytecode-module-output-format=flatbuffer-text %s | IreeFileCheck %s

// CHECK-LABEL: name: "simple_module"
module {
module @simple_module {
// CHECK: exported_functions:
// CHECK: local_name: "func"

// CHECK: internal_functions:
// CHECK: local_name: "func"
func @func(%arg0 : i32) -> i32 attributes { iree.module.export } {
  return %arg0 : i32
}

// CHECK: function_descriptors:
// CHECK-NEXT: bytecode_offset: 0
// CHECK-NEXT: bytecode_length: 3
// CHECK-NEXT: i32_register_count: 1
// CHECK-NEXT: ref_register_count: 0
// CHECK: bytecode_data: [ 84, 1, 0 ]

}
}
