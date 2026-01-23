// RUN: iree-opt --split-input-file --verify-diagnostics --iree-util-verify-initialization-order %s | FileCheck %s

// Valid: Basic module with correct initialization order.

util.global private @global1 = 1 : i32
util.global private @global2 : i32
util.initializer {
  %val = util.global.load @global1 : i32
  util.global.store %val, @global2 : i32
  util.return
}

// -----

// Valid: Immutable global initialized by initial value only.

util.global private @immutable_with_value = 42 : i32

// -----

// Valid: Immutable global initialized by store in initializer only.

util.global private @immutable_no_value : i32
util.initializer {
  %c42 = arith.constant 42 : i32
  util.global.store %c42, @immutable_no_value : i32
  util.return
}

// -----

// Valid: Mutable global can be stored from anywhere.

util.global private mutable @mutable_global : i32
util.initializer {
  %c1 = arith.constant 1 : i32
  util.global.store %c1, @mutable_global : i32
  util.return
}
util.func public @store_mutable() {
  %c2 = arith.constant 2 : i32
  util.global.store %c2, @mutable_global : i32
  util.return
}

// -----

// Valid: Store to immutable global in initializer-only function.

util.func private @init_only_func() {
  %c42 = arith.constant 42 : i32
  util.global.store %c42, @immutable_in_func : i32
  util.return
}

util.global private @immutable_in_func : i32
util.initializer {
  util.call @init_only_func() : () -> ()
  util.return
}

// -----

// Valid: Complex transitive case through multiple functions.

util.func private @leaf_func() {
  %val = util.global.load @early_global : i32
  util.global.store %val, @late_global : i32
  util.return
}

util.func private @middle_func() {
  util.call @leaf_func() : () -> ()
  util.return
}

util.global private @early_global = 1 : i32
util.global private @late_global : i32

util.initializer {
  util.call @middle_func() : () -> ()
  util.return
}

// -----

// Valid: Multiple initializers in correct order.

util.global private @g1 = 1 : i32
util.global private @g2 : i32
util.initializer {
  %val = util.global.load @g1 : i32
  util.global.store %val, @g2 : i32
  util.return
}
util.global private @g3 : i32
util.initializer {
  %val = util.global.load @g2 : i32
  util.global.store %val, @g3 : i32
  util.return
}

// -----

// Error: Forward reference in initializer.

util.initializer {
  // expected-error @+1 {{initializer at position 0 accesses global '@later_global' defined at position 1}}
  %val = util.global.load @later_global : i32
  util.return
}
util.global private @later_global = 42 : i32

// -----

// Error: Store to forward global in initializer.

util.initializer {
  %c42 = arith.constant 42 : i32
  // expected-error @+1 {{initializer at position 0 stores to global '@forward_store' defined at position 1}}
  util.global.store %c42, @forward_store : i32
  util.return
}
util.global private @forward_store : i32

// -----

// Error: Double initialization of immutable global.

util.global private @double_init = 10 : i32
util.initializer {
  %c20 = arith.constant 20 : i32
  // expected-error @+1 {{immutable global '@double_init' is initialized both by initial value and by store in initializer}}
  util.global.store %c20, @double_init : i32
  util.return
}

// -----

// Error: Global with initial value modified by earlier initializer.

util.initializer {
  %c99 = arith.constant 99 : i32
  // expected-error @+1 {{initializer at position 0 stores to global '@modified_init_value' defined at position 1}}
  util.global.store %c99, @modified_init_value : i32
  util.return
}
util.global private @modified_init_value = 42 : i32

// -----

// Error: Store to immutable global in externally reachable function.

util.func public @public_func() {
  %c42 = arith.constant 42 : i32
  // expected-error @+1 {{store to immutable global '@no_external_store' in function 'public_func' which is reachable from non-initializer contexts}}
  util.global.store %c42, @no_external_store : i32
  util.return
}
util.global private @no_external_store : i32

// -----

// Warning: Conditional store to immutable global in initializer-only function.

util.func private @conditional_store(%cond: i1) {
  scf.if %cond {
    %c42 = arith.constant 42 : i32
    // expected-warning @+1 {{conditional store to immutable global '@cond_store' in initializer-only function 'conditional_store' may indicate complex initialization pattern (verification may be overly conservative)}}
    util.global.store %c42, @cond_store : i32
  }
  util.return
}

util.global private @cond_store : i32
util.initializer {
  %true = arith.constant true
  util.call @conditional_store(%true) : (i1) -> ()
  util.return
}

// -----

// Warning: Store in loop within initializer-only function.

util.func private @loop_store() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %val = arith.index_cast %i : index to i32
    // expected-warning @+1 {{conditional store to immutable global '@loop_store_global' in initializer-only function 'loop_store' may indicate complex initialization pattern (verification may be overly conservative)}}
    util.global.store %val, @loop_store_global : i32
  }
  util.return
}

util.global private @loop_store_global : i32
util.initializer {
  util.call @loop_store() : () -> ()
  util.return
}

// -----

// Valid: Multiple stores in same initializer (last one wins).

util.global private mutable @multi_store : i32
util.initializer {
  %c1 = arith.constant 1 : i32
  util.global.store %c1, @multi_store : i32
  %c2 = arith.constant 2 : i32
  util.global.store %c2, @multi_store : i32
  %c3 = arith.constant 3 : i32
  util.global.store %c3, @multi_store : i32
  util.return
}

// -----

// Error: Store to immutable global in function called by both initializer and external.

util.func public @external_entry() {
  util.call @shared_func() : () -> ()
  util.return
}

util.func private @shared_func() {
  %c42 = arith.constant 42 : i32
  // expected-error @+1 {{store to immutable global '@shared_global' in function 'shared_func' which is reachable from non-initializer contexts}}
  util.global.store %c42, @shared_global : i32
  util.return
}

util.global private @shared_global : i32
util.initializer {
  util.call @shared_func() : () -> ()
  util.return
}

// -----

// Valid: Empty module with no initializers or globals.

util.func public @empty_module() {
  util.return
}

// -----

// Valid: Module with only globals, no initializers.

util.global private @only_global1 = 1 : i32
util.global private @only_global2 = 2 : i32
util.global private mutable @only_global3 : i32

// -----

// Valid: Empty initializer.

util.initializer {
  util.return
}

// -----

// Valid: Nested function calls all initializer-only.

util.func private @deep3() {
  %c3 = arith.constant 3 : i32
  util.global.store %c3, @deep_global : i32
  util.return
}

util.func private @deep2() {
  util.call @deep3() : () -> ()
  util.return
}

util.func private @deep1() {
  util.call @deep2() : () -> ()
  util.return
}

util.global private @deep_global : i32
util.initializer {
  util.call @deep1() : () -> ()
  util.return
}

// -----

// Warning: Store in while loop in initializer-only function.

util.func private @while_store() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %result = scf.while (%arg = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %arg, %c10 : i32
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%arg: i32):
    // expected-warning @+1 {{conditional store to immutable global '@while_global' in initializer-only function 'while_store' may indicate complex initialization pattern (verification may be overly conservative)}}
    util.global.store %arg, @while_global : i32
    %c1 = arith.constant 1 : i32
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  util.return
}

util.global private @while_global : i32
util.initializer {
  util.call @while_store() : () -> ()
  util.return
}

// -----

// Warning: Store in index_switch in initializer-only function.

util.func private @switch_store(%idx: index) {
  scf.index_switch %idx
  case 0 {
    %c10 = arith.constant 10 : i32
    util.global.store %c10, @switch_global : i32
    scf.yield
  }
  case 1 {
    %c20 = arith.constant 20 : i32
    util.global.store %c20, @switch_global : i32
    scf.yield
  }
  default {
    %c30 = arith.constant 30 : i32
    util.global.store %c30, @switch_global : i32
  }
  util.return
}

util.global private mutable @switch_global : i32
util.initializer {
  %c0 = arith.constant 0 : index
  util.call @switch_store(%c0) : (index) -> ()
  util.return
}

// -----

// Error: Initializer calls function that accesses forward-referenced global.

util.func private @access_forward_global() {
  // expected-error @+1 {{initializer at position 2 transitively accesses global '@forward_global' defined at position 3 through function call}}
  %val = util.global.load @forward_global : i32
  util.return
}

util.global private @early_defined = 1 : i32
util.initializer {
  util.call @access_forward_global() : () -> ()
  util.return
}
util.global private @forward_global : i32

// -----

// Error: Transitive forward reference through multiple function calls.

util.func private @deep_access() {
  // expected-error @+2 {{initializer at position 4 transitively stores to global '@deep_forward_global' defined at position 5 through function call}}
  %c42 = arith.constant 42 : i32
  util.global.store %c42, @deep_forward_global : i32
  util.return
}

util.func private @middle_caller() {
  util.call @deep_access() : () -> ()
  util.return
}

util.global private @base_global = 1 : i32
util.global private @another_global = 2 : i32
util.initializer {
  util.call @middle_caller() : () -> ()
  util.return
}
util.global private @deep_forward_global : i32

// -----

// Valid: Recursive function calls (visited set prevents infinite loop).

util.func private @recursive_func(%depth: i32) {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi sgt, %depth, %c0 : i32
  scf.if %cmp {
    %c1 = arith.constant 1 : i32
    %new_depth = arith.subi %depth, %c1 : i32
    util.call @recursive_func(%new_depth) : (i32) -> ()
  }
  // CHECK: util.global.store %{{.+}}, @recursive_global
  util.global.store %depth, @recursive_global : i32
  util.return
}

util.global private @recursive_global : i32
util.initializer {
  %c5 = arith.constant 5 : i32
  util.call @recursive_func(%c5) : (i32) -> ()
  util.return
}

// -----

// Valid: Mutually recursive functions (visited set handles cycles).

util.func private @func_a(%n: i32) {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi sgt, %n, %c0 : i32
  scf.if %cmp {
    %c1 = arith.constant 1 : i32
    %new_n = arith.subi %n, %c1 : i32
    util.call @func_b(%new_n) : (i32) -> ()
  }
  util.return
}

util.func private @func_b(%n: i32) {
  // CHECK: util.global.load @cycle_global
  %val = util.global.load @cycle_global : i32
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi sgt, %n, %c0 : i32
  scf.if %cmp {
    util.call @func_a(%n) : (i32) -> ()
  }
  util.return
}

util.global private @cycle_global = 42 : i32
util.initializer {
  %c3 = arith.constant 3 : i32
  util.call @func_a(%c3) : (i32) -> ()
  util.return
}

// -----

// Valid: Function called by multiple initializers at different positions.

util.func private @shared_reader() {
  // CHECK: util.global.load @early_shared
  %val = util.global.load @early_shared : i32
  util.return
}

util.global private @early_shared = 1 : i32
util.initializer {
  util.call @shared_reader() : () -> ()
  util.return
}

util.global private @middle_global = 2 : i32
util.initializer {
  util.call @shared_reader() : () -> ()
  util.return
}

// -----

// Valid: Function accesses multiple globals, all defined before first initializer.

util.func private @multi_access() {
  // CHECK: util.global.load @first_global
  %v1 = util.global.load @first_global : i32
  // CHECK: util.global.load @second_global
  %v2 = util.global.load @second_global : i32
  %sum = arith.addi %v1, %v2 : i32
  // CHECK: util.global.store %{{.+}}, @result_global
  util.global.store %sum, @result_global : i32
  util.return
}

util.global private @first_global = 1 : i32
util.global private @second_global = 2 : i32
util.global private @result_global : i32
util.initializer {
  util.call @multi_access() : () -> ()
  util.return
}
