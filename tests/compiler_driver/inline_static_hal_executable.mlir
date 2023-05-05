// RUN: iree-compile \
// RUN:   --compile-to=hal \
// RUN:   --mlir-print-ir-after-all \
// RUN:   --iree-execution-model=inline-static \
// RUN:   --iree-hal-target-backends=vmvx-inline %s | FileCheck %s

func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
  return %0, %arg0 : tensor<4xf32>, tensor<4xf32>
}

// Check the IR not registered as iree_hal_module_register_inline_types
// CHECK-NOT: hal.command_buffer
// CHECK-NOT: hal.allocator
// CHECK-NOT: hal.event
// CHECK-NOT: hal.fence
// CHECK-NOT: hal.pipeline_layout
// CHECK-NOT: hal.semaphore
// CHECK-NOT: hal.executable

// TODO(#12586): Remove this after the issue is fixed.
// XFAIL: *
