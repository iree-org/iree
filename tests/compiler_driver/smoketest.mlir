// RUN: iree-compile --split-input-file \
// RUN:   --iree-hal-target-backends=vmvx \
// RUN:   --iree-vm-bytecode-module-output-format=flatbuffer-text %s | FileCheck %s

// CHECK-LABEL: "name": "simple_module"
module @simple_module {
// CHECK: "exported_functions":
// CHECK: "local_name": "func"

// CHECK: "function_descriptors":
// CHECK-NEXT: {
// CHECK-NEXT:   "bytecode_offset": 0
// CHECK-NEXT:   "bytecode_length": 6
// CHECK-NEXT:   "requirements": 0
// CHECK-NEXT:   "reserved": 0
// CHECK-NEXT:   "block_count": 1
// CHECK-NEXT:   "i32_register_count": 1
// CHECK-NEXT:   "ref_register_count": 0
// CHECK-NEXT: }
func.func @func(%arg0 : i32) -> i32 {
  return %arg0 : i32
}

// CHECK: "bytecode_data": [
// CHECK-NEXT:   121,
// CHECK-NEXT:   90,
// CHECK-NEXT:   1,
// CHECK-NEXT:   0,
// CHECK-NEXT:   0,
// CHECK-NEXT:   0,
}

// -----

// CHECK-LABEL: "name": "do_not_optimize_module"
module @do_not_optimize_module {
// CHECK: "exported_functions":
// CHECK: "local_name": "add"
func.func @add() -> i32 {
  %c1 = arith.constant 1 : i32
  %unf_c1 = util.optimization_barrier %c1 : i32
  %unf_c2 = util.unfoldable_constant 2 : i32
  %result = arith.addi %unf_c1, %unf_c2 : i32
  return %result : i32
}
}

// -----

// CHECK-LABEL: "name": "hal_usage"
module @hal_usage {
// CHECK: "imported_functions":
// CHECK: "full_name": "hal.command_buffer.dispatch"
// CHECK: "exported_functions":
// CHECK: "local_name": "hloElementwiseOps"
// CHECK: "local_name": "__init"
func.func @hloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  %1 = arith.subf %0, %arg0 : tensor<4xf32>
  %2 = arith.mulf %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
}
