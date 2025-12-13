// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(vm.module(iree-vm-annotate-functions))" \
// RUN:   %s | FileCheck %s

// vm.yield op in function body sets vm.yield attribute.

// CHECK-LABEL: @yield_op_direct
vm.module @yield_op_direct {
  // CHECK: vm.func private @yields_directly() attributes {vm.yield}
  vm.func private @yields_directly() {
    vm.yield ^done
  ^done:
    vm.return
  }
}

// -----

// vm.yield propagates from function with vm.yield op to callers.

// CHECK-LABEL: @yield_op_propagates
vm.module @yield_op_propagates {
  // CHECK: vm.func private @yields_directly() attributes {vm.yield}
  vm.func private @yields_directly() {
    vm.yield ^done
  ^done:
    vm.return
  }
  // CHECK: vm.func private @caller() attributes {vm.yield}
  vm.func private @caller() {
    vm.call @yields_directly() : () -> ()
    vm.return
  }
}

// -----

// vm.yield propagates from import to caller.

// CHECK-LABEL: @yield_from_import
vm.module @yield_from_import {
  vm.import private @yielding() attributes {vm.yield}
  // CHECK: vm.func private @caller() attributes {vm.yield}
  vm.func private @caller() {
    vm.call @yielding() : () -> ()
    vm.return
  }
}

// -----

// vm.yield propagates transitively through call graph.

// CHECK-LABEL: @yield_transitive
vm.module @yield_transitive {
  vm.import private @yielding() attributes {vm.yield}
  // CHECK: vm.func private @level1() attributes {vm.yield}
  vm.func private @level1() {
    vm.call @yielding() : () -> ()
    vm.return
  }
  // CHECK: vm.func private @level2() attributes {vm.yield}
  vm.func private @level2() {
    vm.call @level1() : () -> ()
    vm.return
  }
  // CHECK: vm.func private @level3() attributes {vm.yield}
  vm.func private @level3() {
    vm.call @level2() : () -> ()
    vm.return
  }
}

// -----

// Function with refs and failable ops (allocation) gets vm.unwind.

// CHECK-LABEL: @unwind_refs_failable
vm.module @unwind_refs_failable {
  // CHECK: vm.func private @needs_unwind() attributes {vm.unwind}
  vm.func private @needs_unwind() {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    %buf = vm.buffer.alloc %c128, %c16 : !vm.buffer
    vm.return
  }
}

// -----

// Function with refs but NO failable ops does NOT get vm.unwind.

// CHECK-LABEL: @no_unwind_refs_only
vm.module @no_unwind_refs_only {
  // CHECK: vm.func private @no_unwind
  // CHECK-NOT: vm.unwind
  vm.func private @no_unwind(%r: !vm.ref<?>) {
    %nz = vm.cmp.nz.ref %r : !vm.ref<?>
    vm.return
  }
}

// -----

// Function with failable ops but NO refs does NOT get vm.unwind.

// CHECK-LABEL: @no_unwind_failable_only
vm.module @no_unwind_failable_only {
  vm.import private @failable()
  // CHECK: vm.func private @no_unwind
  // CHECK-NOT: vm.unwind
  vm.func private @no_unwind() {
    vm.call @failable() : () -> ()
    vm.return
  }
}

// -----

// Pure integer function gets neither attribute.

// CHECK-LABEL: @no_attributes
vm.module @no_attributes {
  // CHECK: vm.func private @pure_int
  // CHECK-NOT: vm.yield
  // CHECK-NOT: vm.unwind
  vm.func private @pure_int(%val: i32) -> i32 {
    %one = vm.const.i32 1
    %result = vm.add.i32 %val, %one : i32
    vm.return %result : i32
  }
}

// -----

// vm.unwind propagates from callee to caller.

// CHECK-LABEL: @unwind_propagates
vm.module @unwind_propagates {
  // Leaf has refs and failable ops (buffer.alloc) -> needs unwind.
  // CHECK: vm.func private @leaf() attributes {vm.unwind}
  vm.func private @leaf() {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    %buf = vm.buffer.alloc %c128, %c16 : !vm.buffer
    vm.return
  }
  // Caller inherits unwind from leaf.
  // CHECK: vm.func private @caller() attributes {vm.unwind}
  vm.func private @caller() {
    vm.call @leaf() : () -> ()
    vm.return
  }
}

// -----

// vm.unwind from import propagates to caller.

// CHECK-LABEL: @unwind_from_import
vm.module @unwind_from_import {
  vm.import private @unwindable() attributes {vm.unwind}
  // CHECK: vm.func private @caller() attributes {vm.unwind}
  vm.func private @caller() {
    vm.call @unwindable() : () -> ()
    vm.return
  }
}

// -----

// Both vm.yield and vm.unwind can be set on the same function.

// CHECK-LABEL: @both_yield_and_unwind
vm.module @both_yield_and_unwind {
  vm.import private @yielding_unwindable() attributes {vm.yield, vm.unwind}
  // CHECK: vm.func private @caller() attributes {vm.unwind, vm.yield}
  vm.func private @caller() {
    vm.call @yielding_unwindable() : () -> ()
    vm.return
  }
}

// -----

// Refs in return type trigger unwind if there are failable ops.

// CHECK-LABEL: @ref_return_with_failable
vm.module @ref_return_with_failable {
  vm.import private @create_buffer() -> !vm.buffer
  // CHECK: vm.func private @returns_ref() -> !vm.buffer attributes {vm.unwind}
  vm.func private @returns_ref() -> !vm.buffer {
    %buf = vm.call @create_buffer() : () -> !vm.buffer
    vm.return %buf : !vm.buffer
  }
}

// -----

// Explicit vm.fail op is failable.

// CHECK-LABEL: @fail_op_is_failable
vm.module @fail_op_is_failable {
  // CHECK: vm.func private @may_fail(%{{.*}}: !vm.buffer) attributes {vm.unwind}
  vm.func private @may_fail(%buf: !vm.buffer) {
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    vm.cond_br %nz, ^ok, ^fail
  ^ok:
    vm.return
  ^fail:
    %code = vm.const.i32 1
    vm.fail %code, "buffer was null"
  }
}

// -----

// vm.cond_fail is failable.

// CHECK-LABEL: @cond_fail_is_failable
vm.module @cond_fail_is_failable {
  // CHECK: vm.func private @may_cond_fail(%{{.*}}: !vm.buffer) attributes {vm.unwind}
  vm.func private @may_cond_fail(%buf: !vm.buffer) {
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    %code = vm.const.i32 1
    vm.cond_fail %nz, %code, "buffer was null"
    vm.return
  }
}

// -----

// SCC: mutual recursion shares attributes.

// CHECK-LABEL: @scc_mutual_recursion
vm.module @scc_mutual_recursion {
  vm.import private @yielding() attributes {vm.yield}
  // Both functions in the cycle should get vm.yield.
  // CHECK: vm.func private @func_a() attributes {vm.yield}
  vm.func private @func_a() {
    vm.call @func_b() : () -> ()
    vm.return
  }
  // CHECK: vm.func private @func_b() attributes {vm.yield}
  vm.func private @func_b() {
    vm.call @yielding() : () -> ()
    vm.call @func_a() : () -> ()
    vm.return
  }
}

// -----

// SCC: self-recursion with vm.unwind.

// CHECK-LABEL: @scc_self_recursion
vm.module @scc_self_recursion {
  // CHECK: vm.func private @recursive() attributes {vm.unwind}
  vm.func private @recursive() {
    %c128 = vm.const.i64 128
    %c16 = vm.const.i32 16
    %buf = vm.buffer.alloc %c128, %c16 : !vm.buffer
    %len = vm.buffer.length %buf : !vm.buffer -> i64
    %zero = vm.const.i64 0
    %cmp = vm.cmp.gt.i64.s %len, %zero : i64
    vm.cond_br %cmp, ^recurse, ^done
  ^recurse:
    vm.call @recursive() : () -> ()
    vm.br ^done
  ^done:
    vm.return
  }
}

// -----

// Variadic call propagates attributes.

// CHECK-LABEL: @variadic_call
vm.module @variadic_call {
  vm.import private @variadic_yielding(%arg: i32 ...) attributes {vm.yield}
  // CHECK: vm.func private @caller() attributes {vm.yield}
  vm.func private @caller() {
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    vm.call.variadic @variadic_yielding([%c1, %c2]) : (i32 ...) -> ()
    vm.return
  }
}

// -----

// Buffer load/store ops have MayFail trait.

// CHECK-LABEL: @buffer_load_store_may_fail
vm.module @buffer_load_store_may_fail {
  // CHECK: vm.func private @buffer_ops(%{{.*}}: !vm.buffer) attributes {vm.unwind}
  vm.func private @buffer_ops(%buf: !vm.buffer) {
    %c0 = vm.const.i64 0
    %val = vm.buffer.load.i32 %buf[%c0] : !vm.buffer -> i32
    vm.buffer.store.i32 %val, %buf[%c0] : i32 -> !vm.buffer
    vm.return
  }
}

// -----

// List ops have MayFail trait.

// CHECK-LABEL: @list_ops_may_fail
vm.module @list_ops_may_fail {
  // CHECK: vm.func private @list_ops() attributes {vm.unwind}
  vm.func private @list_ops() {
    %c10 = vm.const.i32 10
    %list = vm.list.alloc %c10 : (i32) -> !vm.list<i32>
    %c0 = vm.const.i32 0
    %val = vm.list.get.i32 %list, %c0 : (!vm.list<i32>, i32) -> i32
    vm.return
  }
}
