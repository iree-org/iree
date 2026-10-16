// Mutable globals require an initializer that allocates their backing buffer
// with hal_inline ops; compiling through the VM lowering requires inlining
// that initializer into the combined initializer.

// RUN: iree-compile \
// RUN:   --compile-to=vm \
// RUN:   --iree-execution-model=inline-dynamic \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx %s | \
// RUN: FileCheck %s --check-prefix=CHECK-DYNAMIC

// CHECK-DYNAMIC-LABEL: vm.func private @__init()
// CHECK-DYNAMIC-DAG: vm.call @hal_inline.buffer.allocate
// CHECK-DYNAMIC-DAG: vm.global.store.ref %{{.+}}, @state
// CHECK-DYNAMIC-DAG: vm.call @hal_loader.executable.load
// CHECK-DYNAMIC: vm.global.store.ref %{{.+}}, @update_dispatch_0

// RUN: iree-compile \
// RUN:   --compile-to=vm \
// RUN:   --iree-execution-model=inline-static \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx-inline %s | \
// RUN: FileCheck %s --check-prefix=CHECK-STATIC

// CHECK-STATIC-LABEL: vm.func private @__init()
// CHECK-STATIC: vm.call @hal_inline.buffer.allocate
// CHECK-STATIC: vm.global.store.ref %{{.+}}, @state

util.global private mutable @state = dense<1.0> : tensor<4xf32>

func.func @update(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %state = util.global.load @state : tensor<4xf32>
  %sum = arith.addf %state, %arg0 : tensor<4xf32>
  util.global.store %sum, @state : tensor<4xf32>
  return %sum : tensor<4xf32>
}
