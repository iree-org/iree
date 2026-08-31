// RUN: iree-compile \
// RUN:   --compile-to=hal \
// RUN:   --iree-execution-model=inline-dynamic \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s

func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
  return %0, %arg0 : tensor<4xf32>, tensor<4xf32>
}

// Check that the IR isn't using types from the full HAL
// (only those in iree_hal_module_register_loader_types).
// CHECK-NOT: hal.command_buffer
// CHECK-NOT: hal.allocator
// CHECK-NOT: hal.event
// CHECK-NOT: hal.fence
// CHECK-NOT: hal.pipeline_layout
// CHECK-NOT: hal.semaphore
// CHECK-NOT: hal.executable private

// Compiling through the VM lowering requires inlining the executable-loading
// initializer into the combined initializer.
// RUN: iree-compile \
// RUN:   --compile-to=vm \
// RUN:   --iree-execution-model=inline-dynamic \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=CHECK-VM

// CHECK-VM-LABEL: vm.func private @__init()
// CHECK-VM: vm.call @hal_loader.executable.query_support
// CHECK-VM: vm.call @hal_loader.executable.load
// CHECK-VM: vm.global.store.ref %{{.+}}, @simple_mul_dispatch_0
