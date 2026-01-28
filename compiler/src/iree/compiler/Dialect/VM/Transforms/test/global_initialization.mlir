// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-global-initialization))" %s | FileCheck %s

// Tests an empty module is initialized empty.

// CHECK: vm.module public @init_empty {
// CHECK: vm.func private @__init()
// CHECK-NEXT: vm.return
// CHECK: vm.func private @__deinit()
// CHECK-NEXT: vm.return
// CHECK: }
vm.module @init_empty {
}

// -----

// CHECK-LABEL: @init_i32
vm.module @init_i32 {
  // CHECK: vm.global.i32 private @g0
  vm.global.i32 private @g0 : i32 = 0 : i32

  // CHECK: vm.global.i32 private mutable @g1 : i32
  vm.global.i32 private mutable @g1 = 123 : i32

  // CHECK: vm.global.i32 private mutable @g2 : i32
  vm.global.i32 private @g2 = 123 : i32

  // CHECK: vm.func private @__init() {
  // CHECK-NEXT:   %c123 = vm.const.i32 123
  // CHECK-NEXT:   vm.global.store.i32 %c123, @g1
  // CHECK-NEXT:   %c123_0 = vm.const.i32 123
  // CHECK-NEXT:   vm.global.store.i32 %c123_0, @g2
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @mutability_change
vm.module @mutability_change {
  // CHECK: vm.global.i32 private @g0
  vm.global.i32 private mutable @g0 : i32 = 0 : i32
  // CHECK: vm.global.i32 private mutable @g1 : i32
  vm.global.i32 private mutable @g1 = 123 : i32
  // CHECK: vm.global.i32 private mutable @g2 : i32
  vm.global.i32 private @g2 : i32
  // CHECK: vm.global.i64 private mutable @g3
  vm.global.i64 private @g3 = 12345 : i64

  vm.initializer {
    %c456 = vm.const.i32 456
    vm.global.store.i32 %c456, @g2 : i32
    vm.return
  }

  // CHECK: vm.func public @func
  vm.func public @func() {
    // CHECK: vm.global.load.i32 immutable @g0
    vm.global.load.i32 @g0 : i32
    // CHECK: vm.global.load.i32 @g1
    vm.global.load.i32 @g1 : i32
    // CHECK: vm.global.load.i32 @g2
    vm.global.load.i32 immutable @g2 : i32
    // CHECK: vm.global.load.i64 @g3
    vm.global.load.i64 immutable @g3 : i64
    vm.return
  }

  // CHECK: vm.func private @__init() {
  // CHECK-NEXT:   %c123 = vm.const.i32 123
  // CHECK-NEXT:   vm.global.store.i32 %c123, @g1
  // CHECK-NEXT:   %c12345 = vm.const.i64 12345
  // CHECK-NEXT:   vm.global.store.i64 %c12345, @g3
  // CHECK-NEXT:   %c456 = vm.const.i32 456
  // CHECK-NEXT:   vm.global.store.i32 %c456, @g2
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @init_ref
vm.module @init_ref {
  // CHECK: vm.global.ref private mutable @g0 : !vm.ref<?>
  vm.global.ref private mutable @g0 : !vm.ref<?>

  // CHECK: vm.global.ref private mutable @g1 : !vm.ref<?>
  vm.global.ref private mutable @g1 : !vm.ref<?>

  // CHECK: vm.global.ref private @g2 : !vm.ref<?>
  vm.global.ref private @g2 : !vm.ref<?>

  // CHECK: vm.func private @__init()
  // CHECK-NEXT: vm.return
}

// -----

// CHECK-LABEL: @initializers
vm.module @initializers {
  // CHECK: vm.global.i32 private mutable @g0 : i32
  vm.global.i32 private @g0 : i32
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %c123 = vm.const.i32 123
    vm.global.store.i32 %c123, @g0 : i32
    vm.return
  }

  // CHECK: vm.global.ref private mutable @g1 : !vm.ref<?>
  vm.global.ref private mutable @g1 : !vm.ref<?>
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %null = vm.const.ref.zero : !vm.ref<?>
    vm.global.store.ref %null, @g1 : !vm.ref<?>
    vm.return
  }

  // CHECK: vm.global.ref private mutable @g2 : !vm.ref<?>
  vm.global.ref private mutable @g2 : !vm.ref<?>
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %g1 = vm.global.load.ref @g1 : !vm.ref<?>
    vm.global.store.ref %g1, @g2 : !vm.ref<?>
    vm.return
  }

  //      CHECK: vm.export @__init
  // CHECK-NEXT: vm.func private @__init() {
  // CHECK-NEXT:   %c123 = vm.const.i32 123
  // CHECK-NEXT:   vm.global.store.i32 %c123, @g0 : i32
  // CHECK-NEXT:   %null = vm.const.ref.zero : !vm.ref<?>
  // CHECK-NEXT:   vm.global.store.ref %null, @g1 : !vm.ref<?>
  // CHECK-NEXT:   %g1 = vm.global.load.ref @g1 : !vm.ref<?>
  // CHECK-NEXT:   vm.global.store.ref %g1, @g2 : !vm.ref<?>
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @unused_globals
vm.module @unused_globals {
  // CHECK: vm.global.i32 private mutable @used
  vm.global.i32 private @used : i32 = 1 : i32
  // CHECK-NOT: vm.global.i32 private @unused
  vm.global.i32 private @unused : i32 = 2 : i32
  vm.func @foo() {
    %0 = vm.global.load.i32 @used : i32
    vm.return
  }
}

// -----

// Tests that the pass is safe to run if there's already an empty __init.

// CHECK-LABEL: @existing_empty_init
vm.module @existing_empty_init {
  // A new inline initializer that needs to run.
  // CHECK: vm.global.i32 private mutable @g0 : i32
  vm.global.i32 private @g0 = 8 : i32

  // CHECK: vm.export @__init
  vm.export @__init
  // CHECK-NEXT: vm.func private @__init()
  vm.func private @__init() {
    vm.return

    // New initializers are merged in at the end.
    // CHECK-NEXT: vm.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: %c8 = vm.const.i32 8
    // CHECK-NEXT: vm.global.store.i32 %c8, @g0 : i32
    // CHECK-NEXT: vm.return
  }
}

// -----

// Tests that the pass is safe to run if there's already been an __init added.

// CHECK-LABEL: @existing_init
vm.module @existing_init {
  // A new inline initializer that needs to run.
  // CHECK: vm.global.i32 private mutable @g0 : i32
  vm.global.i32 private @g0 = 8 : i32

  // New initializer that needs to run.
  // CHECK: vm.global.i32 private mutable @g1 : i32
  vm.global.i32 private @g1 : i32
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %c123 = vm.const.i32 123
    vm.global.store.i32 %c123, @g1 : i32
    vm.return
  }

  // Handled in __init already. No new code should be added.
  // CHECK: vm.global.ref private mutable @g2 : !vm.ref<?>
  vm.global.ref private mutable @g2 : !vm.ref<?>

  // CHECK: vm.export @__init
  vm.export @__init
  // CHECK-NEXT: vm.func private @__init()
  vm.func private @__init() {
    // These existing contents will remain at the top.
    // CHECK-NEXT: %[[NULL:.+]] = vm.const.ref.zero
    %null = vm.const.ref.zero : !vm.ref<?>
    // CHECK-NEXT: vm.global.store.ref %[[NULL]], @g2
    vm.global.store.ref %null, @g2 : !vm.ref<?>

    // To test that we can handle CFGs we have this dummy branch.
    // CHECK: %[[COND:.+]] = vm.const.i32 1
    %cond = vm.const.i32 1
    // CHECK: vm.cond_br %[[COND]], ^bb1, ^bb2
    vm.cond_br %cond, ^bb1, ^bb2

  // CHECK: ^bb1:
  ^bb1:
    // CHECK-NEXT: vm.br ^bb3
    vm.return

  // CHECK: ^bb2:
  ^bb2:
    // CHECK-NEXT: vm.br ^bb3
    vm.return

    // New initializers are merged in at the end.
    // CHECK: ^bb3:
    // CHECK-NEXT: %c8 = vm.const.i32 8
    // CHECK-NEXT: vm.global.store.i32 %c8, @g0 : i32
    // CHECK-NEXT: %c123 = vm.const.i32 123
    // CHECK-NEXT: vm.global.store.i32 %c123, @g1 : i32
    // CHECK-NEXT: vm.return
  }
}

// -----

// Tests that initializers can reference globals even when the initializer
// appears before the global in module order. This is a regression test for
// a bug where GlobalInitializationPass processed ops in module order,
// causing initializers to run before globals were initialized.

// CHECK-LABEL: @initializer_before_global
vm.module @initializer_before_global {
  // Initializer appears first in module order but should run AFTER @g0 is initialized.
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %g0 = vm.global.load.i32 @g0 : i32
    vm.global.store.i32 %g0, @result : i32
    vm.return
  }

  // Global with initial value appears second in module order.
  // CHECK: vm.global.i32 private mutable @g0 : i32
  vm.global.i32 private @g0 = 42 : i32

  // Result global
  // CHECK: vm.global.i32 private mutable @result : i32
  vm.global.i32 private @result : i32

  // CHECK: vm.export @__init
  // CHECK-NEXT: vm.func private @__init() {
  // Phase 1: Initialize globals with initial values
  // CHECK-NEXT:   %c42 = vm.const.i32 42
  // CHECK-NEXT:   vm.global.store.i32 %c42, @g0
  // Phase 2: Execute initializers in module order
  // CHECK-NEXT:   %[[G0:.+]] = vm.global.load.i32 @g0
  // CHECK-NEXT:   vm.global.store.i32 %[[G0]], @result
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// -----

// Tests complex interleaving of globals and initializers.

// CHECK-LABEL: @mixed_order
vm.module @mixed_order {
  // Initializer 1 (references g1 which appears later)
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %g1 = vm.global.load.i32 @g1 : i32
    vm.global.store.i32 %g1, @result1 : i32
    vm.return
  }

  // Global 1 (referenced by initializer 1)
  // CHECK: vm.global.i32 private mutable @g1 : i32
  vm.global.i32 private @g1 = 100 : i32

  // Global 2 (no initial value)
  // CHECK: vm.global.i32 private mutable @g2 : i32
  vm.global.i32 private @g2 : i32

  // Initializer 2 (stores to g2)
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %c200 = vm.const.i32 200
    vm.global.store.i32 %c200, @g2 : i32
    vm.return
  }

  // Initializer 3 (loads g2 set by initializer 2)
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %g2 = vm.global.load.i32 @g2 : i32
    vm.global.store.i32 %g2, @result2 : i32
    vm.return
  }

  // Result globals
  // CHECK: vm.global.i32 private mutable @result1 : i32
  vm.global.i32 private @result1 : i32
  // CHECK: vm.global.i32 private mutable @result2 : i32
  vm.global.i32 private @result2 : i32

  // CHECK: vm.export @__init
  // CHECK-NEXT: vm.func private @__init() {
  // Phase 1: Initialize globals with initial values
  // CHECK-NEXT:   %c100 = vm.const.i32 100
  // CHECK-NEXT:   vm.global.store.i32 %c100, @g1
  // Phase 2: Execute initializers in module order
  // CHECK-NEXT:   %[[G1:.+]] = vm.global.load.i32 @g1
  // CHECK-NEXT:   vm.global.store.i32 %[[G1]], @result1
  // CHECK-NEXT:   %c200 = vm.const.i32 200
  // CHECK-NEXT:   vm.global.store.i32 %c200, @g2
  // CHECK-NEXT:   %[[G2:.+]] = vm.global.load.i32 @g2
  // CHECK-NEXT:   vm.global.store.i32 %[[G2]], @result2
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}
