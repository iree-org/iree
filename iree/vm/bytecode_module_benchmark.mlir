vm.module @bytecode_module_benchmark {
  // Measures the pure overhead of calling into/returning from a module.
  vm.export @empty_func
  vm.func @empty_func() {
    vm.return
  }

  // Measures the cost of a call an internal function.
  vm.func @internal_func(%arg0 : i32) -> i32 attributes {noinline} {
    vm.return %arg0 : i32
  }
  vm.export @call_internal_func
  vm.func @call_internal_func(%arg0 : i32) -> i32 {
    %0 = vm.call @internal_func(%arg0) : (i32) -> i32
    %1 = vm.call @internal_func(%0) : (i32) -> i32
    %2 = vm.call @internal_func(%1) : (i32) -> i32
    %3 = vm.call @internal_func(%2) : (i32) -> i32
    %4 = vm.call @internal_func(%3) : (i32) -> i32
    %5 = vm.call @internal_func(%4) : (i32) -> i32
    %6 = vm.call @internal_func(%5) : (i32) -> i32
    %7 = vm.call @internal_func(%6) : (i32) -> i32
    %8 = vm.call @internal_func(%7) : (i32) -> i32
    %9 = vm.call @internal_func(%8) : (i32) -> i32
    %10 = vm.call @internal_func(%9) : (i32) -> i32
    %11 = vm.call @internal_func(%10) : (i32) -> i32
    %12 = vm.call @internal_func(%11) : (i32) -> i32
    %13 = vm.call @internal_func(%12) : (i32) -> i32
    %14 = vm.call @internal_func(%13) : (i32) -> i32
    %15 = vm.call @internal_func(%14) : (i32) -> i32
    %16 = vm.call @internal_func(%15) : (i32) -> i32
    %17 = vm.call @internal_func(%16) : (i32) -> i32
    %18 = vm.call @internal_func(%17) : (i32) -> i32
    %19 = vm.call @internal_func(%18) : (i32) -> i32
    %20 = vm.call @internal_func(%19) : (i32) -> i32
    vm.return %20 : i32
  }

  // Measures the cost of a call to an imported function.
  vm.import @benchmark.imported_func(%arg : i32) -> i32
  vm.export @call_imported_func
  vm.func @call_imported_func(%arg0 : i32) -> i32 {
    %0 = vm.call @benchmark.imported_func(%arg0) : (i32) -> i32
    %1 = vm.call @benchmark.imported_func(%0) : (i32) -> i32
    %2 = vm.call @benchmark.imported_func(%1) : (i32) -> i32
    %3 = vm.call @benchmark.imported_func(%2) : (i32) -> i32
    %4 = vm.call @benchmark.imported_func(%3) : (i32) -> i32
    %5 = vm.call @benchmark.imported_func(%4) : (i32) -> i32
    %6 = vm.call @benchmark.imported_func(%5) : (i32) -> i32
    %7 = vm.call @benchmark.imported_func(%6) : (i32) -> i32
    %8 = vm.call @benchmark.imported_func(%7) : (i32) -> i32
    %9 = vm.call @benchmark.imported_func(%8) : (i32) -> i32
    %10 = vm.call @benchmark.imported_func(%9) : (i32) -> i32
    %11 = vm.call @benchmark.imported_func(%10) : (i32) -> i32
    %12 = vm.call @benchmark.imported_func(%11) : (i32) -> i32
    %13 = vm.call @benchmark.imported_func(%12) : (i32) -> i32
    %14 = vm.call @benchmark.imported_func(%13) : (i32) -> i32
    %15 = vm.call @benchmark.imported_func(%14) : (i32) -> i32
    %16 = vm.call @benchmark.imported_func(%15) : (i32) -> i32
    %17 = vm.call @benchmark.imported_func(%16) : (i32) -> i32
    %18 = vm.call @benchmark.imported_func(%17) : (i32) -> i32
    %19 = vm.call @benchmark.imported_func(%18) : (i32) -> i32
    %20 = vm.call @benchmark.imported_func(%19) : (i32) -> i32
    vm.return %20 : i32
  }

  // Measures the cost of a simple for-loop.
  vm.export @loop_sum
  vm.func @loop_sum(%count : i32) -> i32 {
    %c1 = vm.const.i32 1 : i32
    %i0 = vm.const.i32.zero : i32
    vm.br ^loop(%i0 : i32)
  ^loop(%i : i32):
    %in = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %in, %count : i32
    vm.cond_br %cmp, ^loop(%in : i32), ^loop_exit(%in : i32)
  ^loop_exit(%ie : i32):
    vm.return %ie : i32
  }
}
