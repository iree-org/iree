vm.module @async_ops {
  //===--------------------------------------------------------------------===//
  // vm.yield
  //===--------------------------------------------------------------------===//

  // Tests a simple straight-line yield sequence that requires 3 resumes.
  // Starts with 100, adds 1 three times across yields, expects 103.
  vm.export @test_yield_sequence
  vm.func @test_yield_sequence() {
    %c1 = vm.const.i32 1
    %c100 = vm.const.i32 100
    %c100_dno = vm.optimization_barrier %c100 : i32
    %y0 = vm.add.i32 %c100_dno, %c1 : i32
    %y0_dno = vm.optimization_barrier %y0 : i32
    vm.yield ^bb1
  ^bb1:
    %y1 = vm.add.i32 %y0_dno, %c1 : i32
    %y1_dno = vm.optimization_barrier %y1 : i32
    vm.yield ^bb2
  ^bb2:
    %y2 = vm.add.i32 %y1_dno, %c1 : i32
    %y2_dno = vm.optimization_barrier %y2 : i32
    vm.yield ^bb3
  ^bb3:
    %c103 = vm.const.i32 103
    vm.check.eq %y2_dno, %c103, "100+1+1+1=103" : i32
    vm.return
  }

  // Tests a yield with data-dependent control flow (true branch).
  vm.export @test_yield_divergent_true
  vm.func @test_yield_divergent_true() {
    %c1 = vm.const.i32 1
    %c100 = vm.const.i32 100
    %c200 = vm.const.i32 200
    %cond = vm.cmp.nz.i32 %c1 : i32
    vm.cond_br %cond, ^true, ^false
  ^true:
    %v_true = vm.optimization_barrier %c100 : i32
    vm.yield ^check(%v_true : i32)
  ^false:
    %v_false = vm.optimization_barrier %c200 : i32
    vm.yield ^check(%v_false : i32)
  ^check(%result : i32):
    vm.check.eq %result, %c100, "cond=1 selects true branch" : i32
    vm.return
  }

  // Tests a yield with data-dependent control flow (false branch).
  vm.export @test_yield_divergent_false
  vm.func @test_yield_divergent_false() {
    %c0 = vm.const.i32 0
    %c100 = vm.const.i32 100
    %c200 = vm.const.i32 200
    %cond = vm.cmp.nz.i32 %c0 : i32
    vm.cond_br %cond, ^true, ^false
  ^true:
    %v_true = vm.optimization_barrier %c100 : i32
    vm.yield ^check(%v_true : i32)
  ^false:
    %v_false = vm.optimization_barrier %c200 : i32
    vm.yield ^check(%v_false : i32)
  ^check(%result : i32):
    vm.check.eq %result, %c200, "cond=0 selects false branch" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.call.yieldable with internal functions
  //===--------------------------------------------------------------------===//

  // Internal function that yields 4 times, incrementing a counter each time.
  // Takes a starting value and returns starting + 4.
  vm.func private @yield_counter(%start : i32) -> i32 {
    %c1 = vm.const.i32 1
    %v0 = vm.add.i32 %start, %c1 : i32
    %v0_dno = vm.optimization_barrier %v0 : i32
    vm.yield ^y1
  ^y1:
    %v1 = vm.add.i32 %v0_dno, %c1 : i32
    %v1_dno = vm.optimization_barrier %v1 : i32
    vm.yield ^y2
  ^y2:
    %v2 = vm.add.i32 %v1_dno, %c1 : i32
    %v2_dno = vm.optimization_barrier %v2 : i32
    vm.yield ^y3
  ^y3:
    %v3 = vm.add.i32 %v2_dno, %c1 : i32
    vm.return %v3 : i32
  }

  // Tests calling an internal yieldable function.
  // The callee yields 4 times. Expects result of 0 + 4 = 4.
  vm.export @test_call_yieldable_internal attributes {emitc.exclude}
  vm.func @test_call_yieldable_internal() {
    %c0 = vm.const.i32 0
    vm.call.yieldable @yield_counter(%c0) : (i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c4 = vm.const.i32 4
    vm.check.eq %result, %c4, "0+4=4" : i32
    vm.return
  }

  // Internal function that takes an input and yields once, returning input + 1.
  vm.func private @yield_add_one(%arg0: i32) -> i32 {
    %c1 = vm.const.i32 1
    %result = vm.add.i32 %arg0, %c1 : i32
    %result_dno = vm.optimization_barrier %result : i32
    vm.yield ^done
  ^done:
    vm.return %result_dno : i32
  }

  // Tests calling an internal yieldable function with an argument.
  // Expects result of 42 + 1 = 43.
  vm.export @test_call_yieldable_with_arg attributes {emitc.exclude}
  vm.func @test_call_yieldable_with_arg() {
    %c42 = vm.const.i32 42
    vm.call.yieldable @yield_add_one(%c42) : (i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c43 = vm.const.i32 43
    vm.check.eq %result, %c43, "42+1=43" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.call.yieldable with imported functions
  //===--------------------------------------------------------------------===//

  // Import from native yieldable_test module.
  // yield_n(arg: i32, yield_count: i32) -> i32
  // Yields yield_count times, returns arg + yield_count.
  vm.import private @yieldable_test.yield_n(%arg : i32, %yield_count : i32) -> i32 attributes {vm.yield}

  // yield_variadic_sum(args: i32..., yield_count: i32) -> i32
  // Sums all variadic i32 args, yields yield_count times, returns sum + yield_count.
  vm.import private @yieldable_test.yield_variadic_sum(%args : i32 ..., %yield_count : i32) -> i32 attributes {vm.yield}

  // Test: call yieldable import with 3 yields.
  // Expects 100 + 3 = 103.
  vm.export @test_call_yieldable_import_yields_3 attributes {emitc.exclude}
  vm.func @test_call_yieldable_import_yields_3() {
    %c100 = vm.const.i32 100
    %c3 = vm.const.i32 3
    vm.call.yieldable @yieldable_test.yield_n(%c100, %c3) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c103 = vm.const.i32 103
    vm.check.eq %result, %c103, "100+3=103" : i32
    vm.return
  }

  // Test: call yieldable import with 0 yields (synchronous).
  // Expects immediate return with 42.
  vm.export @test_call_yieldable_import_yields_0 attributes {emitc.exclude}
  vm.func @test_call_yieldable_import_yields_0() {
    %c42 = vm.const.i32 42
    %c0 = vm.const.i32 0
    vm.call.yieldable @yieldable_test.yield_n(%c42, %c0) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.check.eq %result, %c42, "42+0=42" : i32
    vm.return
  }

  // Test: call yieldable import after internal function call.
  // This exercises return_registers clearing after internal call.
  vm.func private @internal_add_10(%x : i32) -> i32 {
    %c10 = vm.const.i32 10
    %r = vm.add.i32 %x, %c10 : i32
    vm.return %r : i32
  }

  // Expects (5 + 10) + 2 = 17.
  vm.export @test_call_yieldable_after_internal attributes {emitc.exclude}
  vm.func @test_call_yieldable_after_internal() {
    %c5 = vm.const.i32 5
    // First call an internal function (sets return_registers).
    %v1 = vm.call @internal_add_10(%c5) : (i32) -> i32
    // Then call yieldable import.
    %c2 = vm.const.i32 2
    vm.call.yieldable @yieldable_test.yield_n(%v1, %c2) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c17 = vm.const.i32 17
    vm.check.eq %result, %c17, "(5+10)+2=17" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Additional regression tests for frame tracking bugs
  //===--------------------------------------------------------------------===//

  // Test: two sequential yieldable import calls in the same function.
  // Expects (10 + 2) + 3 = 15.
  vm.export @test_call_yieldable_import_sequential attributes {emitc.exclude}
  vm.func @test_call_yieldable_import_sequential() {
    %c10 = vm.const.i32 10
    %c2 = vm.const.i32 2
    %c3 = vm.const.i32 3
    // First yieldable import: yields 2 times, returns 10 + 2 = 12
    vm.call.yieldable @yieldable_test.yield_n(%c10, %c2) : (i32, i32) -> ^after_first(i32)
  ^after_first(%v1 : i32):
    // Second yieldable import: yields 3 times, returns 12 + 3 = 15
    vm.call.yieldable @yieldable_test.yield_n(%v1, %c3) : (i32, i32) -> ^done(i32)
  ^done(%result : i32):
    %c15 = vm.const.i32 15
    vm.check.eq %result, %c15, "(10+2)+3=15" : i32
    vm.return
  }

  // Test: yieldable import nested inside an internal yieldable function.
  // The internal function yields before and after calling the import.
  vm.func private @yield_then_import_then_yield(%arg0 : i32) -> i32 {
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    // Add 1 before yield
    %v0 = vm.add.i32 %arg0, %c1 : i32
    %v0_dno = vm.optimization_barrier %v0 : i32
    vm.yield ^after_first_yield
  ^after_first_yield:
    // Call yieldable import (yields 2 times)
    vm.call.yieldable @yieldable_test.yield_n(%v0_dno, %c2) : (i32, i32) -> ^after_import(i32)
  ^after_import(%v1 : i32):
    // Add 1 after import
    %v2 = vm.add.i32 %v1, %c1 : i32
    %v2_dno = vm.optimization_barrier %v2 : i32
    vm.yield ^final
  ^final:
    vm.return %v2_dno : i32
  }

  // Expects ((50 + 1) + 2) + 1 = 54.
  vm.export @test_call_nested_yieldable attributes {emitc.exclude}
  vm.func @test_call_nested_yieldable() {
    %c50 = vm.const.i32 50
    vm.call.yieldable @yield_then_import_then_yield(%c50) : (i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c54 = vm.const.i32 54
    vm.check.eq %result, %c54, "((50+1)+2)+1=54" : i32
    vm.return
  }

  // Test: stress test with many yields to catch state accumulation bugs.
  // Expects 1000 + 10 = 1010.
  vm.export @test_call_yieldable_import_stress attributes {emitc.exclude}
  vm.func @test_call_yieldable_import_stress() {
    %c1000 = vm.const.i32 1000
    %c10 = vm.const.i32 10
    vm.call.yieldable @yieldable_test.yield_n(%c1000, %c10) : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c1010 = vm.const.i32 1010
    vm.check.eq %result, %c1010, "1000+10=1010" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.call.variadic.yieldable with imported functions
  //===--------------------------------------------------------------------===//

  // Test: call variadic yieldable import with 2 args and 3 yields.
  // Expects (10 + 20) + 3 = 33.
  vm.export @test_call_variadic_yieldable_2args attributes {emitc.exclude}
  vm.func @test_call_variadic_yieldable_2args() {
    %c10 = vm.const.i32 10
    %c20 = vm.const.i32 20
    %c3 = vm.const.i32 3
    vm.call.variadic.yieldable @yieldable_test.yield_variadic_sum(%c10, %c20, %c3) {segment_sizes = dense<[2, 1]> : vector<2xi16>, segment_types = [i32, i32]} : (i32, i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c33 = vm.const.i32 33
    vm.check.eq %result, %c33, "(10+20)+3=33" : i32
    vm.return
  }

  // Test: call variadic yieldable import with 0 yields (synchronous).
  // Expects 5 + 10 + 15 = 30.
  vm.export @test_call_variadic_yieldable_0yields attributes {emitc.exclude}
  vm.func @test_call_variadic_yieldable_0yields() {
    %c5 = vm.const.i32 5
    %c10 = vm.const.i32 10
    %c15 = vm.const.i32 15
    %c0 = vm.const.i32 0
    vm.call.variadic.yieldable @yieldable_test.yield_variadic_sum(%c5, %c10, %c15, %c0) {segment_sizes = dense<[3, 1]> : vector<2xi16>, segment_types = [i32, i32]} : (i32, i32, i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c30 = vm.const.i32 30
    vm.check.eq %result, %c30, "5+10+15+0=30" : i32
    vm.return
  }

  // Test: call variadic yieldable import with single arg.
  // Expects 100 + 2 = 102.
  vm.export @test_call_variadic_yieldable_1arg attributes {emitc.exclude}
  vm.func @test_call_variadic_yieldable_1arg() {
    %c100 = vm.const.i32 100
    %c2 = vm.const.i32 2
    vm.call.variadic.yieldable @yieldable_test.yield_variadic_sum(%c100, %c2) {segment_sizes = dense<[1, 1]> : vector<2xi16>, segment_types = [i32, i32]} : (i32, i32) -> ^resume(i32)
  ^resume(%result : i32):
    %c102 = vm.const.i32 102
    vm.check.eq %result, %c102, "100+2=102" : i32
    vm.return
  }

  // Test: call variadic yieldable import with empty variadic list.
  // Expects 0 + 1 = 1.
  vm.export @test_call_variadic_yieldable_empty attributes {emitc.exclude}
  vm.func @test_call_variadic_yieldable_empty() {
    %c1 = vm.const.i32 1
    vm.call.variadic.yieldable @yieldable_test.yield_variadic_sum(%c1) {segment_sizes = dense<[0, 1]> : vector<2xi16>, segment_types = [i32, i32]} : (i32) -> ^resume(i32)
  ^resume(%result : i32):
    vm.check.eq %result, %c1, "0+1=1" : i32
    vm.return
  }

  // Test: two sequential variadic yieldable calls.
  // Expects ((10 + 20) + 2) + (32 + 5) + 1 = 38.
  vm.export @test_call_variadic_yieldable_sequential attributes {emitc.exclude}
  vm.func @test_call_variadic_yieldable_sequential() {
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    %c5 = vm.const.i32 5
    %c10 = vm.const.i32 10
    %c20 = vm.const.i32 20
    // First variadic yieldable: sum(10, 20) + 2 yields = 32
    vm.call.variadic.yieldable @yieldable_test.yield_variadic_sum(%c10, %c20, %c2) {segment_sizes = dense<[2, 1]> : vector<2xi16>, segment_types = [i32, i32]} : (i32, i32, i32) -> ^after_first(i32)
  ^after_first(%v1 : i32):
    // Second variadic yieldable: sum(32, 5) + 1 yield = 38
    vm.call.variadic.yieldable @yieldable_test.yield_variadic_sum(%v1, %c5, %c1) {segment_sizes = dense<[2, 1]> : vector<2xi16>, segment_types = [i32, i32]} : (i32, i32, i32) -> ^done(i32)
  ^done(%result : i32):
    %c38 = vm.const.i32 38
    vm.check.eq %result, %c38, "((10+20)+2)+((32+5)+1)=38" : i32
    vm.return
  }
}
