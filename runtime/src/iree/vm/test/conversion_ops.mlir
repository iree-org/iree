vm.module @conversion_ops {

  //===----------------------------------------------------------------------===//
  // Casting and type conversion/emulation
  //===----------------------------------------------------------------------===//

  vm.export @test_trunc_i32_i8
  vm.func private @test_trunc_i32_i8() {
    %c1 = vm.const.i32 2147483647
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.trunc.i32.i8 %c1dno : i32 -> i32
    %c2 = vm.const.i32 255
    vm.check.eq %v, %c2, "truncate unsigned i32 to unsigned i8" : i32
    vm.return
  }

  vm.export @test_trunc_i32_i16
  vm.func private @test_trunc_i32_i16() {
    %c1 = vm.const.i32 2147483647
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.trunc.i32.i16 %c1dno : i32 -> i32
    %c2 = vm.const.i32 65535
    vm.check.eq %v, %c2, "truncate unsigned i32 to unsigned i16" : i32
    vm.return
  }

  vm.export @test_cast_any_ref attributes {emitc.exclude}
  vm.func private @test_cast_any_ref() {
    %c128 = vm.const.i64 128
    %alignment = vm.const.i32 16
    %buffer = vm.buffer.alloc %c128, %alignment : !vm.buffer
    %any = vm.cast.ref.any %buffer : !vm.buffer -> !vm.ref<?>
    %any_dno = util.optimization_barrier %any : !vm.ref<?>
    %cast = vm.cast.any.ref %any_dno : !vm.ref<?> -> !vm.buffer
    vm.check.eq %buffer, %cast, "cast should succeed" : !vm.buffer
    vm.return
  }

  vm.export @test_cast_any_ref_null attributes {emitc.exclude}
  vm.func private @test_cast_any_ref_null() {
    %null = vm.const.ref.zero : !vm.buffer
    %any = vm.cast.ref.any %null : !vm.buffer -> !vm.ref<?>
    %any_dno = util.optimization_barrier %any : !vm.ref<?>
    %cast = vm.cast.any.ref %any_dno : !vm.ref<?> -> !vm.buffer
    vm.check.eq %null, %cast, "cast should succeed on nulls" : !vm.buffer
    vm.return
  }

  vm.export @fail_cast_any_ref attributes {emitc.exclude}
  vm.func private @fail_cast_any_ref() {
    %c128 = vm.const.i64 128
    %alignment = vm.const.i32 16
    %buffer = vm.buffer.alloc %c128, %alignment : !vm.buffer
    %any = vm.cast.ref.any %buffer : !vm.buffer -> !vm.ref<?>
    %any_dno = util.optimization_barrier %any : !vm.ref<?>
    // Should fail at runtime because of the type mismatch.
    %cast = vm.cast.any.ref %any_dno : !vm.ref<?> -> !vm.list<?>
    util.optimization_barrier %cast : !vm.list<?>
    vm.return
  }

}
