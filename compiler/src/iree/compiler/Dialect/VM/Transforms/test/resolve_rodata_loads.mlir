// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-util-fold-globals,vm.module(iree-vm-resolve-rodata-loads,symbol-dce))" --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @rodata_ref
module {
  vm.module public @rodata_ref {
    vm.rodata private @_const_0 dense<[1, 0]> : tensor<2xi32>
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[VALUE:.+]] = vm.const.ref.rodata @_const_0 : !vm.buffer
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[VALUE]]
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
// CHECK-LABEL: @rodata_ref_multiple_uniform
module {
  vm.module public @rodata_ref_multiple_uniform {
    vm.rodata private @_const_0 dense<[1, 0]> : tensor<2xi32>
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[VALUE:.+]] = vm.const.ref.rodata @_const_0 : !vm.buffer
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[VALUE]]
      vm.return %0 : !vm.buffer
    }
    // CHECK-NOT: vm.global.ref
    // CHECK-NOT: vm.global.store.ref
    vm.global.ref private @__constant_1x2xi32 : !vm.buffer
    vm.initializer {
      %rodata = vm.const.ref.rodata @_const_0 : !vm.buffer
      vm.global.store.ref %rodata, @__constant_1x2xi32 : !vm.buffer
      vm.return
    }
    // CHECK-NOT: vm.global.store.ref
    vm.initializer {
      // After rodata deduplication we can end up with multiple initializers
      // doing the same thing.
      %rodata = vm.const.ref.rodata @_const_0 : !vm.buffer
      vm.global.store.ref %rodata, @__constant_1x2xi32 : !vm.buffer
      vm.return
    }
  }
}

// -----
// CHECK-LABEL: @rodata_ref_multiple_nonuniform
module {
  vm.module public @rodata_ref_multiple_nonuniform {
    vm.rodata private @_const_0 dense<[1, 0]> : tensor<2xi32>
    vm.rodata private @_const_1 dense<[0, 1]> : tensor<2xi32>
    // CHECK: vm.func public @f
    vm.func public @f() -> !vm.buffer {
      // CHECK: %[[VALUE:.+]] = vm.global.load.ref @__constant_1x2xi32
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[VALUE]]
      vm.return %0 : !vm.buffer
    }
    // CHECK: vm.global.ref private @__constant_1x2xi32
    vm.global.ref private @__constant_1x2xi32 : !vm.buffer
    vm.initializer {
      // CHECK: %[[VALUE0:.+]] = vm.const.ref.rodata @_const_0
      %rodata = vm.const.ref.rodata @_const_0 : !vm.buffer
      // CHECK: vm.global.store.ref %[[VALUE0]], @__constant_1x2xi32
      vm.global.store.ref %rodata, @__constant_1x2xi32 : !vm.buffer
      vm.return
    }
    vm.initializer {
      // CHECK: %[[VALUE1:.+]] = vm.const.ref.rodata @_const_1
      %rodata = vm.const.ref.rodata @_const_1 : !vm.buffer
      // CHECK: vm.global.store.ref %[[VALUE1]], @__constant_1x2xi32
      vm.global.store.ref %rodata, @__constant_1x2xi32 : !vm.buffer
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
      // CHECK: %[[VALUE:.+]] = vm.const.ref.zero : !vm.buffer
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[VALUE]]
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
      // CHECK: %[[VALUE:.+]] = vm.global.load.ref @__constant_1x2xi32
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[VALUE]]
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
      // CHECK: %[[VALUE:.+]] = vm.global.load.ref
      %0 = vm.global.load.ref @__constant_1x2xi32 : !vm.buffer
      // CHECK: vm.return %[[VALUE]]
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
