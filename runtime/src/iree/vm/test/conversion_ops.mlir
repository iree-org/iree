vm.module @conversion_ops {

  //===----------------------------------------------------------------------===//
  // Casting and type conversion/emulation
  //===----------------------------------------------------------------------===//

  vm.export @test_trunc_i32_i8
  vm.func @test_trunc_i32_i8() {
    %c1 = vm.const.i32 2147483647
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.trunc.i32.i8 %c1dno : i32 -> i32
    %c2 = vm.const.i32 255
    vm.check.eq %v, %c2, "truncate unsigned i32 to unsigned i8" : i32
    vm.return
  }

  vm.export @test_trunc_i32_i16
  vm.func @test_trunc_i32_i16() {
    %c1 = vm.const.i32 2147483647
    %c1dno = util.optimization_barrier %c1 : i32
    %v = vm.trunc.i32.i16 %c1dno : i32 -> i32
    %c2 = vm.const.i32 65535
    vm.check.eq %v, %c2, "truncate unsigned i32 to unsigned i16" : i32
    vm.return
  }

}
