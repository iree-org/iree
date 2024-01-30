vm.module @bytecode_module_benchmark {
  // Measures the pure overhead of calling into/returning from a module.
  vm.export @empty_func
  vm.func @empty_func() {
    vm.return
  }

  // Measures the cost of a call an internal function.
  vm.func @internal_func(%arg0 : i32) -> i32 attributes {inlining_policy = #util.inline.never} {
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
  vm.import private @native_import_module.add_1(%arg : i32) -> i32
  vm.export @call_imported_func
  vm.func @call_imported_func(%arg0 : i32) -> i32 {
    %0 = vm.call @native_import_module.add_1(%arg0) : (i32) -> i32
    %1 = vm.call @native_import_module.add_1(%0) : (i32) -> i32
    %2 = vm.call @native_import_module.add_1(%1) : (i32) -> i32
    %3 = vm.call @native_import_module.add_1(%2) : (i32) -> i32
    %4 = vm.call @native_import_module.add_1(%3) : (i32) -> i32
    %5 = vm.call @native_import_module.add_1(%4) : (i32) -> i32
    %6 = vm.call @native_import_module.add_1(%5) : (i32) -> i32
    %7 = vm.call @native_import_module.add_1(%6) : (i32) -> i32
    %8 = vm.call @native_import_module.add_1(%7) : (i32) -> i32
    %9 = vm.call @native_import_module.add_1(%8) : (i32) -> i32
    %10 = vm.call @native_import_module.add_1(%9) : (i32) -> i32
    %11 = vm.call @native_import_module.add_1(%10) : (i32) -> i32
    %12 = vm.call @native_import_module.add_1(%11) : (i32) -> i32
    %13 = vm.call @native_import_module.add_1(%12) : (i32) -> i32
    %14 = vm.call @native_import_module.add_1(%13) : (i32) -> i32
    %15 = vm.call @native_import_module.add_1(%14) : (i32) -> i32
    %16 = vm.call @native_import_module.add_1(%15) : (i32) -> i32
    %17 = vm.call @native_import_module.add_1(%16) : (i32) -> i32
    %18 = vm.call @native_import_module.add_1(%17) : (i32) -> i32
    %19 = vm.call @native_import_module.add_1(%18) : (i32) -> i32
    %20 = vm.call @native_import_module.add_1(%19) : (i32) -> i32
    vm.return %20 : i32
  }

  // Measures the cost of a simple for-loop.
  vm.export @loop_sum
  vm.func @loop_sum(%count : i32) -> i32 {
    %c1 = vm.const.i32 1
    %i0 = vm.const.i32.zero
    vm.br ^loop(%i0 : i32)
  ^loop(%i : i32):
    %in = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %in, %count : i32
    vm.cond_br %cmp, ^loop(%in : i32), ^loop_exit(%in : i32)
  ^loop_exit(%ie : i32):
    vm.return %ie : i32
  }

  // Measures the cost of lots of buffer loads.
  vm.export @buffer_reduce
  vm.func @buffer_reduce(%count : i32) -> i32 {
    %c0 = vm.const.i64.zero
    %c0_i32 = vm.const.i32.zero
    %pattern = vm.const.i32 1
    %c1 = vm.const.i64 1
    %c4 = vm.const.i64 4
    %count_i64 = vm.ext.i32.i64.u %count : i32 -> i64
    %count_bytes = vm.mul.i64 %count_i64, %c4 : i64
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %count_bytes, %alignment : !vm.buffer
    vm.buffer.fill.i32 %buf, %c0, %count_i64, %pattern : i32 -> !vm.buffer
    vm.br ^loop(%c0, %c0_i32 : i64, i32)
  ^loop(%i : i64, %sum : i32):
    %element = vm.buffer.load.i32 %buf[%i] : !vm.buffer -> i32
    %new_sum = vm.add.i32 %sum, %element : i32
    %ip1 = vm.add.i64 %i, %c1 : i64
    %cmp = vm.cmp.lt.i64.s %ip1, %count_i64 : i64
    vm.cond_br %cmp, ^loop(%ip1, %new_sum : i64, i32), ^loop_exit(%new_sum : i32)
  ^loop_exit(%result : i32):
    vm.return %result : i32
  }

  // Measures the cost of lots of buffer loads when somewhat unrolled.
  // NOTE: unrolled 8x, requires %count to be % 8 = 0.
  vm.export @buffer_reduce_unrolled
  vm.func @buffer_reduce_unrolled(%count : i32) -> i32 {
    %c0 = vm.const.i64.zero
    %pattern = vm.const.i32 1
    %c1 = vm.const.i64 1
    %c4 = vm.const.i64 4
    %count_i64 = vm.ext.i32.i64.u %count : i32 -> i64
    %count_bytes = vm.mul.i64 %count_i64, %c4 : i64
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %count_bytes, %alignment : !vm.buffer
    vm.buffer.fill.i32 %buf, %c0, %count_i64, %pattern : i32 -> !vm.buffer
    %sum_init = vm.const.i32.zero
    vm.br ^loop(%c0, %sum_init : i64, i32)
  ^loop(%i0 : i64, %sum : i32):
    // TODO(#5544): add addressing modes to load/store.
    %e0 = vm.buffer.load.i32 %buf[%i0] : !vm.buffer -> i32
    %i1 = vm.add.i64 %i0, %c1 : i64
    %e1 = vm.buffer.load.i32 %buf[%i1] : !vm.buffer -> i32
    %i2 = vm.add.i64 %i1, %c1 : i64
    %e2 = vm.buffer.load.i32 %buf[%i2] : !vm.buffer -> i32
    %i3 = vm.add.i64 %i2, %c1 : i64
    %e3 = vm.buffer.load.i32 %buf[%i3] : !vm.buffer -> i32
    %i4 = vm.add.i64 %i3, %c1 : i64
    %e4 = vm.buffer.load.i32 %buf[%i4] : !vm.buffer -> i32
    %i5 = vm.add.i64 %i4, %c1 : i64
    %e5 = vm.buffer.load.i32 %buf[%i5] : !vm.buffer -> i32
    %i6 = vm.add.i64 %i5, %c1 : i64
    %e6 = vm.buffer.load.i32 %buf[%i6] : !vm.buffer -> i32
    %i7 = vm.add.i64 %i6, %c1 : i64
    %e7 = vm.buffer.load.i32 %buf[%i7] : !vm.buffer -> i32
    // If we do reductions like this we could add a horizontal-add op.
    %new_sum0 = vm.add.i32 %sum, %e0 : i32
    %new_sum1 = vm.add.i32 %new_sum0, %e1 : i32
    %new_sum2 = vm.add.i32 %new_sum1, %e2 : i32
    %new_sum3 = vm.add.i32 %new_sum2, %e3 : i32
    %new_sum4 = vm.add.i32 %new_sum3, %e4 : i32
    %new_sum5 = vm.add.i32 %new_sum4, %e5 : i32
    %new_sum6 = vm.add.i32 %new_sum5, %e6 : i32
    %new_sum7 = vm.add.i32 %new_sum6, %e7 : i32
    %next_i = vm.add.i64 %i7, %c1 : i64
    %cmp = vm.cmp.lt.i64.s %next_i, %count_i64 : i64
    vm.cond_br %cmp, ^loop(%next_i, %new_sum7 : i64, i32), ^loop_exit(%new_sum7 : i32)
  ^loop_exit(%result : i32):
    vm.return %result : i32
  }
}
