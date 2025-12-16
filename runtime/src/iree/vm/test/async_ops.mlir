// Tested by iree/vm/bytecode/dispatch_async_test.cc.

vm.module @async_ops {
  //===--------------------------------------------------------------------===//
  // vm.yield
  //===--------------------------------------------------------------------===//

  // Tests a simple straight-line yield sequence that requires 3 resumes.
  //
  // Expects a result of %arg0 + 3.
  vm.export @yield_sequence
  vm.func @yield_sequence(%arg0: i32) -> i32 {
    %c1 = vm.const.i32 1
    %y0 = vm.add.i32 %arg0, %c1 : i32
    %y0_dno = util.optimization_barrier %y0 : i32
    vm.yield ^bb1
  ^bb1:
    %y1 = vm.add.i32 %y0_dno, %c1 : i32
    %y1_dno = util.optimization_barrier %y1 : i32
    vm.yield ^bb2
  ^bb2:
    %y2 = vm.add.i32 %y1_dno, %c1 : i32
    %y2_dno = util.optimization_barrier %y2 : i32
    vm.yield ^bb3
  ^bb3:
    vm.return %y2_dno : i32
  }

  // Tests a yield with data-dependent control, ensuring that we run the
  // alternating branches and pass along branch args on resume.
  //
  // Expects a result of %arg0 ? %arg1 : %arg2.
  vm.export @yield_divergent
  vm.func @yield_divergent(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %cond = vm.cmp.nz.i32 %arg0 : i32
    vm.cond_br %cond, ^true, ^false
  ^true:
    %arg1_dno = util.optimization_barrier %arg1 : i32
    vm.yield ^bb3(%arg1_dno : i32)
  ^false:
    %arg2_dno = util.optimization_barrier %arg2 : i32
    vm.yield ^bb3(%arg2_dno: i32)
  ^bb3(%result : i32):
    vm.return %result : i32
  }

  //===--------------------------------------------------------------------===//
  // vm.call.yieldable with internal functions
  //===--------------------------------------------------------------------===//

  // Internal function that yields 4 times, incrementing a counter each time.
  // Takes a starting value and returns starting + 4.
  vm.func private @yield_counter(%start : i32) -> i32 {
    %c1 = vm.const.i32 1
    %v0 = vm.add.i32 %start, %c1 : i32
    %v0_dno = util.optimization_barrier %v0 : i32
    vm.yield ^y1
  ^y1:
    %v1 = vm.add.i32 %v0_dno, %c1 : i32
    %v1_dno = util.optimization_barrier %v1 : i32
    vm.yield ^y2
  ^y2:
    %v2 = vm.add.i32 %v1_dno, %c1 : i32
    %v2_dno = util.optimization_barrier %v2 : i32
    vm.yield ^y3
  ^y3:
    %v3 = vm.add.i32 %v2_dno, %c1 : i32
    vm.return %v3 : i32
  }

  // Tests calling an internal yieldable function.
  // The callee yields 4 times, so we need 4 resumes.
  // Expects result of 0 + 4 = 4.
  vm.export @call_yieldable_internal attributes {emitc.exclude}
  vm.func @call_yieldable_internal() -> i32 {
    %c0 = vm.const.i32 0
    vm.call.yieldable @yield_counter(%c0) : (i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.return %result : i32
  }

  // Internal function that takes an input and yields once, returning input + 1.
  vm.func private @yield_add_one(%arg0: i32) -> i32 {
    %c1 = vm.const.i32 1
    %result = vm.add.i32 %arg0, %c1 : i32
    %result_dno = util.optimization_barrier %result : i32
    vm.yield ^done
  ^done:
    vm.return %result_dno : i32
  }

  // Tests calling an internal yieldable function with an argument.
  // Expects result of %arg0 + 1.
  vm.export @call_yieldable_with_arg attributes {emitc.exclude}
  vm.func @call_yieldable_with_arg(%arg0: i32) -> i32 {
    vm.call.yieldable @yield_add_one(%arg0) : (i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.return %result : i32
  }

  //===--------------------------------------------------------------------===//
  // vm.call.yieldable with imported functions
  //===--------------------------------------------------------------------===//

  // Import from native yieldable_test module.
  // yield_n(arg: i32, yield_count: i32) -> i32
  // Yields yield_count times, returns arg + yield_count.
  vm.import private @yieldable_test.yield_n(%arg : i32, %yield_count : i32) -> i32 attributes {vm.yield}

  // Test: call yieldable import with 3 yields.
  // Expected: 3 DEFERRED returns, then OK with result = arg + 3
  vm.export @call_yieldable_import_yields_3 attributes {emitc.exclude}
  vm.func @call_yieldable_import_yields_3(%arg0 : i32) -> i32 {
    %c3 = vm.const.i32 3
    vm.call.yieldable @yieldable_test.yield_n(%arg0, %c3) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.return %result : i32
  }

  // Test: call yieldable import with 0 yields (synchronous).
  // Expected: immediate OK with result = arg
  vm.export @call_yieldable_import_yields_0 attributes {emitc.exclude}
  vm.func @call_yieldable_import_yields_0(%arg0 : i32) -> i32 {
    %c0 = vm.const.i32 0
    vm.call.yieldable @yieldable_test.yield_n(%arg0, %c0) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.return %result : i32
  }

  // Test: call yieldable import after internal function call.
  // This exercises Bug 2 fix: return_registers must be cleared after internal call.
  vm.func private @internal_add_10(%x : i32) -> i32 {
    %c10 = vm.const.i32 10
    %r = vm.add.i32 %x, %c10 : i32
    vm.return %r : i32
  }

  vm.export @call_yieldable_after_internal attributes {emitc.exclude}
  vm.func @call_yieldable_after_internal(%arg0 : i32) -> i32 {
    // First call an internal function (sets return_registers).
    %v1 = vm.call @internal_add_10(%arg0) : (i32) -> i32
    // Then call yieldable import (should see return_registers == NULL for begin).
    %c2 = vm.const.i32 2
    vm.call.yieldable @yieldable_test.yield_n(%v1, %c2) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.return %result : i32
  }

  //===--------------------------------------------------------------------===//
  // Additional regression tests for frame tracking bugs
  //===--------------------------------------------------------------------===//

  // Test: two sequential yieldable import calls in the same function.
  // This catches bugs where the second call sees stale state from the first.
  // Expected: 2 yields from first call + 3 yields from second call = 5 total
  // Result: (arg + 2) + 3 = arg + 5
  vm.export @call_yieldable_import_sequential attributes {emitc.exclude}
  vm.func @call_yieldable_import_sequential(%arg0 : i32) -> i32 {
    %c2 = vm.const.i32 2
    %c3 = vm.const.i32 3
    // First yieldable import: yields 2 times, returns arg + 2
    vm.call.yieldable @yieldable_test.yield_n(%arg0, %c2) : (i32, i32) -> ^after_first(i32)
  ^after_first(%v1 : i32):
    // Second yieldable import: yields 3 times, returns v1 + 3 = arg + 5
    vm.call.yieldable @yieldable_test.yield_n(%v1, %c3) : (i32, i32) -> ^done(i32)
  ^done(%result : i32):
    vm.return %result : i32
  }

  // Test: yieldable import nested inside an internal yieldable function.
  // The internal function yields before and after calling the import.
  // This creates the most complex frame stack scenario.
  vm.func private @yield_then_import_then_yield(%arg0 : i32) -> i32 {
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    // Add 1 before yield
    %v0 = vm.add.i32 %arg0, %c1 : i32
    %v0_dno = util.optimization_barrier %v0 : i32
    vm.yield ^after_first_yield
  ^after_first_yield:
    // Call yieldable import (yields 2 times)
    vm.call.yieldable @yieldable_test.yield_n(%v0_dno, %c2) : (i32, i32) -> ^after_import(i32)
  ^after_import(%v1 : i32):
    // Add 1 after import
    %v2 = vm.add.i32 %v1, %c1 : i32
    %v2_dno = util.optimization_barrier %v2 : i32
    vm.yield ^final
  ^final:
    vm.return %v2_dno : i32
  }

  // Export that calls the nested yieldable function.
  // Expected sequence: 1 yield (internal) + 2 yields (import) + 1 yield (internal) = 4 yields
  // Result: ((arg + 1) + 2) + 1 = arg + 4
  vm.export @call_nested_yieldable attributes {emitc.exclude}
  vm.func @call_nested_yieldable(%arg0 : i32) -> i32 {
    vm.call.yieldable @yield_then_import_then_yield(%arg0) : (i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.return %result : i32
  }

  // Test: stress test with many yields to catch state accumulation bugs.
  // Calls yieldable import with high yield count.
  // Expected: 10 yields, result = arg + 10
  vm.export @call_yieldable_import_stress attributes {emitc.exclude}
  vm.func @call_yieldable_import_stress(%arg0 : i32) -> i32 {
    %c10 = vm.const.i32 10
    vm.call.yieldable @yieldable_test.yield_n(%arg0, %c10) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.return %result : i32
  }
}
