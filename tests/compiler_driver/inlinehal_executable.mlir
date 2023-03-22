// RUN: iree-compile \
// RUN:   --mlir-print-ir-after=inline \
// RUN:   --iree-execution-model=inline-dynamic \
// RUN:   --iree-hal-target-backends=llvm-cpu %s \
// RUN:   --o=/dev/null 2>&1 | FileCheck %s

func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
  return %0, %arg0 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-NOT: hal.command_buffer
// CHECK-NOT: hal.allocator
// CHECK-NOT: hal.event
// CHECK-NOT: hal.fence
// CHECK-NOT: hal.pipeline_layout
// CHECK-NOT: hal.semaphore
// CHECK: vm.module public @module {
