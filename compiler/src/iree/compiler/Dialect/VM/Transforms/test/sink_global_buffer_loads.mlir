// RUN: iree-opt --split-input-file --pass-pipeline="vm.module(iree-vm-sink-global-buffer-loads)" --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @rodata_ref
module {
  vm.module public @rodata_ref {
    vm.rodata private @_const_0 dense<[1, 0]> : tensor<2xi32>
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[V:.*]] = vm.const.ref.rodata @_const_0 : !vm.buffer
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[V]]
      vm.return %0 : !vm.buffer
    }
    // CHECK-NOT: vm.global.ref
    // CHECK-NOT: vm.global.store.ref
    vm.global.ref private @__constant_1x2xi32 : !vm.buffer
    vm.initializer {
      %_const_0 = vm.const.ref.rodata @_const_0 : !vm.buffer
      vm.global.store.ref %_const_0, @__constant_1x2xi32 : !vm.buffer
      vm.return
    }
  }
}

// -----
// CHECK-LABEL: @undefined
module {
  vm.module public @undefined {
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[V:.*]] = vm.const.ref.zero : !vm.buffer
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[V]]
      vm.return %0 : !vm.buffer
    }
    vm.global.ref private @__constant_1x2xi32 : !vm.buffer
  }
}

// -----
// CHECK-LABEL: @unknown_initializer
module {
  vm.module public @unknown_initializer {
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[V:.*]] = vm.global.load.ref @__constant_1x2xi32
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[V]]
      vm.return %0 : !vm.buffer
    }
    vm.global.ref private @__constant_1x2xi32 : !vm.buffer
    // CHECK: vm.initializer
    vm.initializer {
      %_const_0 = "undefined.custom_initializer"() : () -> (!vm.buffer)
      // CHECK: vm.global.store.ref
      vm.global.store.ref %_const_0, @__constant_1x2xi32 : !vm.buffer
      vm.return
    }
  }
}

// -----
// CHECK-LABEL: @const.ref.zero
module {
  vm.module public @const.ref.zero {
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[NULL:.*]] = vm.const.ref.zero : !vm.buffer
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[NULL]]
      vm.return %0 : !vm.buffer
    }
    // CHECK-NOT: vm.global.ref
    // CHECK-NOT: vm.global.store.ref
    vm.global.ref private @__constant_1x2xi32 : !vm.buffer
    vm.initializer {
      %_const_0 = vm.const.ref.zero : !vm.buffer
      vm.global.store.ref %_const_0, @__constant_1x2xi32 : !vm.buffer
      vm.return
    }
  }
}

// -----
// CHECK-LABEL: @mutable_ref
module {
  vm.module public @mutable_ref {
    vm.rodata private @_const_0 dense<[1, 0]> : tensor<2xi32>
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[V:.*]] = vm.global.load.ref
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[V]]
      vm.return %0 : !vm.buffer
    }
    vm.global.ref private mutable @__constant_1x2xi32 : !vm.buffer
    vm.initializer {
      %_const_0 = vm.const.ref.rodata @_const_0 : !vm.buffer
      vm.global.store.ref %_const_0, @__constant_1x2xi32 : !vm.buffer
      vm.return
    }
  }
}
